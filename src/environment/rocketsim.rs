use crate::environment::Environment;
use rocketsim_rs::{
    bytes::ToBytes,
    cxx::UniquePtr,
    sim::{Arena, ArenaConfig, CarConfig, CarControls, GameMode, Team},
};
use std::error::Error;
use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::str::FromStr;
use std::sync::Once;

const RLVISER_PORT: u16 = 45243;
const ROCKETSIM_PORT: u16 = 34254;

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum UdpPacketTypes {
    Quit = 0,
    GameState = 1,
    Connection = 2,
    Paused = 3,
    Speed = 4,
    Render = 5,
}

static INIT_ROCKETSIM: Once = Once::new();

use std::cell::RefCell;

pub struct RocketSimEnvironment {
    arena: RefCell<UniquePtr<Arena>>,
    car_id: u32,
    lookup_table: Vec<CarControls>,
    socket: UdpSocket,
    rlviser_addr: SocketAddr,
    tick_skip: i32,
    prev_dist_to_ball: f32,
    prev_dist_ball_to_net: f32,
}

// Implement Clone is required by our trait but Arena UniquePtr can't be cloned.
// So we will recreate it if clone is called.
impl Clone for RocketSimEnvironment {
    fn clone(&self) -> Self {
        Self::new(self.tick_skip)
    }
}

impl RocketSimEnvironment {
    pub fn new(tick_skip: i32) -> Self {
        INIT_ROCKETSIM.call_once(|| {
            rocketsim_rs::init(None, true);
        });

        let mut arena = Arena::new(GameMode::Soccar, ArenaConfig::default(), 120);
        let car_id = arena.pin_mut().add_car(Team::Blue, CarConfig::octane());

        let lookup_table = Self::make_lookup_table();

        let socket = UdpSocket::bind(("0.0.0.0", 0)).expect("Failed to bind UDP socket");
        socket.set_nonblocking(true).unwrap();

        let rlviser_addr = SocketAddr::new(IpAddr::from_str("127.0.0.1").unwrap(), RLVISER_PORT);
        
        let _ = socket.send_to(&[UdpPacketTypes::Connection as u8], rlviser_addr);

        let mut env = Self {
            arena: RefCell::new(arena),
            car_id,
            lookup_table,
            socket,
            rlviser_addr,
            tick_skip,
            prev_dist_to_ball: 0.0,
            prev_dist_ball_to_net: 0.0,
        };
        
        env.reset().unwrap();
        env
    }

    fn make_lookup_table() -> Vec<CarControls> {
        let mut actions = Vec::new();
        let bins = [-1.0, 0.0, 1.0];

        // Ground
        for throttle in bins {
            for steer in bins {
                for boost in [false, true] {
                    for handbrake in [false, true] {
                        if boost && throttle != 1.0 { continue; }
                        let t = if throttle == 0.0 && boost { 1.0 } else { throttle };
                        actions.push(CarControls {
                            throttle: t as f32,
                            steer: steer as f32,
                            pitch: 0.0,
                            yaw: steer as f32,
                            roll: 0.0,
                            jump: false,
                            boost,
                            handbrake,
                        });
                    }
                }
            }
        }

        // Aerial
        for pitch in bins {
            for yaw in bins {
                for roll in bins {
                    for jump in [false, true] {
                        for boost in [false, true] {
                            if jump && yaw != 0.0 { continue; }
                            if pitch == 0.0 && roll == 0.0 && !jump { continue; }
                            let handbrake = jump && (pitch != 0.0 || yaw != 0.0 || roll != 0.0);
                            actions.push(CarControls {
                                throttle: if boost { 1.0 } else { 0.0 },
                                steer: yaw as f32,
                                pitch: pitch as f32,
                                yaw: yaw as f32,
                                roll: roll as f32,
                                jump,
                                boost,
                                handbrake,
                            });
                        }
                    }
                }
            }
        }
        actions
    }

    fn send_state_to_rlviser(&self) {
        let mut arena = self.arena.borrow_mut();
        let game_state = arena.pin_mut().get_game_state();
        let _ = self.socket.send_to(&[UdpPacketTypes::GameState as u8], self.rlviser_addr);
        let _ = self.socket.send_to(&game_state.to_bytes(), self.rlviser_addr);
    }
}

impl Environment for RocketSimEnvironment {
    fn state_size(&self) -> usize {
        31
    }

    fn action_size(&self) -> usize {
        self.lookup_table.len()
    }

    fn clone_box(&self) -> Box<dyn Environment> {
        Box::new(self.clone())
    }

    fn get_state(&mut self) -> Result<Vec<f64>, Box<dyn Error>> {
        let mut arena = self.arena.borrow_mut();
        let car = arena.pin_mut().get_car(self.car_id);
        let ball = arena.pin_mut().get_ball();

        let mut state = Vec::with_capacity(31);
        
        state.push(ball.pos.x as f64);
        state.push(ball.pos.y as f64);
        state.push(ball.pos.z as f64);
        state.push(ball.vel.x as f64);
        state.push(ball.vel.y as f64);
        state.push(ball.vel.z as f64);
        state.push(ball.ang_vel.x as f64);
        state.push(ball.ang_vel.y as f64);
        state.push(ball.ang_vel.z as f64);
        
        state.push(car.pos.x as f64);
        state.push(car.pos.y as f64);
        state.push(car.pos.z as f64);
        state.push(car.vel.x as f64);
        state.push(car.vel.y as f64);
        state.push(car.vel.z as f64);
        state.push(car.ang_vel.x as f64);
        state.push(car.ang_vel.y as f64);
        state.push(car.ang_vel.z as f64);
        
        state.push(car.rot_mat.forward.x as f64);
        state.push(car.rot_mat.forward.y as f64);
        state.push(car.rot_mat.forward.z as f64);
        state.push(car.rot_mat.right.x as f64);
        state.push(car.rot_mat.right.y as f64);
        state.push(car.rot_mat.right.z as f64);
        state.push(car.rot_mat.up.x as f64);
        state.push(car.rot_mat.up.y as f64);
        state.push(car.rot_mat.up.z as f64);
        
        state.push(car.boost as f64);
        state.push(if car.is_on_ground { 1.0 } else { 0.0 });
        state.push(if car.has_jumped { 1.0 } else { 0.0 });
        state.push(if car.has_double_jumped { 1.0 } else { 0.0 });

        Ok(state)
    }

    fn evaluate_action(&self, _state: &[f64], action_idx: usize) -> f64 {
        let mut arena = self.arena.borrow_mut();
        
        // Save current state
        let original_state = arena.pin_mut().get_game_state();
        let controls = self.lookup_table[action_idx];

        // Ensure we properly unwrap Result to handle errors though evaluate_action returns no error,
        // we'll just ignore if set_car_controls fails here
        let _ = arena.pin_mut().set_car_controls(self.car_id, controls);
        
        let initial_car = arena.pin_mut().get_car(self.car_id);
        let initial_ball = arena.pin_mut().get_ball();
        let initial_dist_to_ball = ((initial_car.pos.x - initial_ball.pos.x).powi(2) + 
                                    (initial_car.pos.y - initial_ball.pos.y).powi(2) + 
                                    (initial_car.pos.z - initial_ball.pos.z).powi(2)).sqrt();

        let goal_pos = rocketsim_rs::math::Vec3::new(0.0, 5120.0, 0.0);
        let initial_dist_ball_to_net = ((initial_ball.pos.x - goal_pos.x).powi(2) + 
                                        (initial_ball.pos.y - goal_pos.y).powi(2) + 
                                        (initial_ball.pos.z - goal_pos.z).powi(2)).sqrt();

        // Step simulation
        arena.pin_mut().step(self.tick_skip as u32);

        // Get new state to calculate reward
        let new_car = arena.pin_mut().get_car(self.car_id);
        let new_ball = arena.pin_mut().get_ball();

        let new_dist_to_ball = ((new_car.pos.x - new_ball.pos.x).powi(2) + 
                                (new_car.pos.y - new_ball.pos.y).powi(2) + 
                                (new_car.pos.z - new_ball.pos.z).powi(2)).sqrt();

        let new_dist_ball_to_net = ((new_ball.pos.x - goal_pos.x).powi(2) + 
                                    (new_ball.pos.y - goal_pos.y).powi(2) + 
                                    (new_ball.pos.z - goal_pos.z).powi(2)).sqrt();

        // Calculate reward
        // Reward for moving closer to the ball
        let ball_dist_reward = initial_dist_to_ball - new_dist_to_ball;

        // Reward for moving ball closer to net
        let net_dist_reward = initial_dist_ball_to_net - new_dist_ball_to_net;
        
        // Touch reward: if distance to ball is less than ~200 units (ball radius 92.75, car bbox is around 118x84x36)
        let touch_reward = if new_dist_to_ball < 200.0 { 10.0 } else { 0.0 };

        let total_reward = ball_dist_reward + (net_dist_reward * 2.0) + touch_reward;

        // Restore original state
        let _ = arena.pin_mut().set_game_state(&original_state);

        total_reward as f64
    }

    fn apply_action(&mut self, action_idx: usize) -> Result<(), Box<dyn Error>> {
        let controls = self.lookup_table[action_idx];
        let mut arena = self.arena.borrow_mut();
        
        arena.pin_mut().set_car_controls(self.car_id, controls)?;

        let car = arena.pin_mut().get_car(self.car_id);
        let ball = arena.pin_mut().get_ball();
        
        self.prev_dist_to_ball = ((car.pos.x - ball.pos.x).powi(2) + 
                                  (car.pos.y - ball.pos.y).powi(2) + 
                                  (car.pos.z - ball.pos.z).powi(2)).sqrt();

        let goal_pos = rocketsim_rs::math::Vec3::new(0.0, 5120.0, 0.0);
        self.prev_dist_ball_to_net = ((ball.pos.x - goal_pos.x).powi(2) + 
                                      (ball.pos.y - goal_pos.y).powi(2) + 
                                      (ball.pos.z - goal_pos.z).powi(2)).sqrt();

        arena.pin_mut().step(self.tick_skip as u32);

        // Required to drop borrow before sending state
        drop(arena);
        self.send_state_to_rlviser();

        Ok(())
    }

    fn reset(&mut self) -> Result<(), Box<dyn Error>> {
        {
            let mut arena = self.arena.borrow_mut();
            arena.pin_mut().reset_to_random_kickoff(None);
        }
        self.send_state_to_rlviser();
        Ok(())
    }
}
