use crate::environment::Environment;
use crate::flat::rocketsim as flat_rs;
use planus::WriteAsOffset;
use rocketsim_rs::{
    cxx::UniquePtr,
    sim::{Arena, ArenaConfig, CarConfig, CarControls, GameMode, Team},
};
use std::error::Error;
use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::str::FromStr;
use std::sync::Once;

const RLVISER_PORT: u16 = 45243;
#[allow(dead_code)]
const ROCKETSIM_PORT: u16 = 34254;

// CommonValues from RLGymSim_CPP/Utils/CommonValues.h
const CAR_MAX_SPEED: f32 = 2300.0;
const BALL_MAX_SPEED: f32 = 6000.0;
const BALL_RADIUS: f32 = 92.75;
const BACK_WALL_Y: f32 = 5120.0;
const BACK_NET_Y: f32 = 6000.0;
const GOAL_HEIGHT: f32 = 642.775;
// Goal back positions used by VelocityBallToGoalReward and AlignBallGoalReward
const ORANGE_GOAL_BACK: (f32, f32, f32) = (0.0, BACK_NET_Y, GOAL_HEIGHT / 2.0);
const BLUE_GOAL_BACK: (f32, f32, f32) = (0.0, -BACK_NET_Y, GOAL_HEIGHT / 2.0);
const GOAL_WIDTH: f32 = 1786.0;

// Reward scaling from main_training.cpp CombinedReward
// multiplier = 100, tick_skip = 8 (C++ value used for weight normalization),
// match_timeout_secs = 25 => match_timeout_ticks = 25 * 120 / 8 = 375
const REWARD_MULTIPLIER: f32 = 100.0;
const MATCH_TIMEOUT_TICKS: f32 = 375.0;

// Approximate distance threshold for ball touch detection (ball radius + car hitbox half-diagonal)
const TOUCH_DIST_THRESHOLD: f32 = 200.0;

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
#[allow(dead_code)]
enum UdpPacketTypes {
    Quit = 0,
    GameState = 1,
    Connection = 2,
    Paused = 3,
    Speed = 4,
    Render = 5,
}

static INIT_ROCKETSIM: Once = Once::new();

static NEXT_ENV_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static VISUALIZER_ID: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);
static LAST_VIS_TIME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

use std::cell::RefCell;

pub struct RocketSimEnvironment {
    arena: RefCell<UniquePtr<Arena>>,
    car_id: u32,
    lookup_table: Vec<CarControls>,
    socket: UdpSocket,
    rlviser_addr: SocketAddr,
    tick_skip: i32,
    first_touch_given: bool,
    builder: RefCell<planus::Builder>,
    id: usize,
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

        let id = NEXT_ENV_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

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
            first_touch_given: false,
            builder: RefCell::new(planus::Builder::new()),
            id,
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
                        if boost && throttle != 1.0 {
                            continue;
                        }
                        let t = if throttle == 0.0 && boost {
                            1.0
                        } else {
                            throttle
                        };
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
                            if jump && yaw != 0.0 {
                                continue;
                            }
                            if pitch == 0.0 && roll == 0.0 && !jump {
                                continue;
                            }
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
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let current_vis = VISUALIZER_ID.load(std::sync::atomic::Ordering::Relaxed);
        let last_time = LAST_VIS_TIME.load(std::sync::atomic::Ordering::Relaxed);

        if current_vis == usize::MAX
            || current_vis == self.id
            || now.saturating_sub(last_time) > 100
        {
            VISUALIZER_ID.store(self.id, std::sync::atomic::Ordering::Relaxed);
            LAST_VIS_TIME.store(now, std::sync::atomic::Ordering::Relaxed);
        } else {
            return;
        }

        let mut arena = self.arena.borrow_mut();
        let game_state = arena.pin_mut().get_game_state();

        let mut builder = self.builder.borrow_mut();
        builder.clear();

        let flat_state = to_flat_game_state(&game_state);

        let packet = flat_rs::Packet {
            message: flat_rs::Message::GameState(Box::new(flat_state)),
        };

        let offset = packet.prepare(&mut builder);
        let bytes = builder.finish(offset, None);

        let len = bytes.len() as u64;
        let mut msg = len.to_be_bytes().to_vec();
        msg.extend_from_slice(bytes);

        let _ = self.socket.send_to(&msg, self.rlviser_addr);
    }
}

impl Environment for RocketSimEnvironment {
    fn state_size(&self) -> usize {
        37
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

        let mut state = Vec::with_capacity(37);

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

        state.push((car.pos.x - ball.pos.x) as f64);
        state.push((car.pos.y - ball.pos.y) as f64);
        state.push((car.pos.z - ball.pos.z) as f64);
        state.push((car.vel.x - ball.vel.x) as f64);
        state.push((car.vel.y - ball.vel.y) as f64);
        state.push((car.vel.z - ball.vel.z) as f64);

        Ok(state)
    }

    fn evaluate_action(&self, _state: &[f64], action_idx: usize) -> f64 {
        let mut arena = self.arena.borrow_mut();

        // Save current state so we can restore after speculative step
        let original_state = arena.pin_mut().get_game_state();
        let controls = self.lookup_table[action_idx];
        let _ = arena.pin_mut().set_car_controls(self.car_id, controls);

        // Capture pre-step ball velocity for HitBallHardReward
        let initial_ball = arena.pin_mut().get_ball();
        let pre_ball_vel = (initial_ball.vel.x, initial_ball.vel.y, initial_ball.vel.z);

        // Advance simulation
        arena.pin_mut().step(self.tick_skip as u32);

        // Read post-step state
        let new_car = arena.pin_mut().get_car(self.car_id);
        let new_ball = arena.pin_mut().get_ball();

        let car_pos = (new_car.pos.x, new_car.pos.y, new_car.pos.z);
        let car_vel = (new_car.vel.x, new_car.vel.y, new_car.vel.z);
        let car_fwd = (
            new_car.rot_mat.forward.x,
            new_car.rot_mat.forward.y,
            new_car.rot_mat.forward.z,
        );
        let ball_pos = (new_ball.pos.x, new_ball.pos.y, new_ball.pos.z);
        let ball_vel = (new_ball.vel.x, new_ball.vel.y, new_ball.vel.z);
        // boost is stored as 0-100 in RocketSim; convert to fraction for reward
        let boost_frac = (new_car.boost / 100.0).clamp(0.0, 1.0);

        let dist_to_ball = v3_len(v3_sub(car_pos, ball_pos));
        let ball_touched = dist_to_ball < TOUCH_DIST_THRESHOLD;

        let mut reward = 0.0f32;

        // --- VelocityBallToGoalReward × 0.64 * M / mtt ---
        // Reward ball velocity component toward the orange goal (Blue team attacks Orange).
        {
            let dir = v3_norm(v3_sub(ORANGE_GOAL_BACK, ball_pos));
            let ball_vel_norm = v3_scale(ball_vel, 1.0 / BALL_MAX_SPEED);
            reward +=
                v3_dot(dir, ball_vel_norm) * 0.64 * REWARD_MULTIPLIER / MATCH_TIMEOUT_TICKS;
        }

        // --- VelocityPlayerToBallReward × 0.64 * M / mtt ---
        // Reward car velocity component toward the ball.
        {
            let dir = v3_norm(v3_sub(ball_pos, car_pos));
            let car_vel_norm = v3_scale(car_vel, 1.0 / CAR_MAX_SPEED);
            reward +=
                v3_dot(dir, car_vel_norm) * 0.64 * REWARD_MULTIPLIER / MATCH_TIMEOUT_TICKS;
        }

        // --- AirReward × 0.05 * M / mtt ---
        // Reward being airborne.
        {
            let r = if new_car.is_on_ground { 0.0 } else { 1.0 };
            reward += r * 0.05 * REWARD_MULTIPLIER / MATCH_TIMEOUT_TICKS;
        }

        // --- FaceBallReward × 0.32 * M / mtt ---
        // Reward car forward vector alignment with direction to ball.
        {
            let dir = v3_norm(v3_sub(ball_pos, car_pos));
            reward += v3_dot(car_fwd, dir) * 0.32 * REWARD_MULTIPLIER / MATCH_TIMEOUT_TICKS;
        }

        // --- TouchBallReward(aerial_weight=2) × 0.25 * M ---
        // Reward touching the ball; higher reward for aerial touches.
        if ball_touched {
            let height_ratio = (new_ball.pos.z + BALL_RADIUS) / (BALL_RADIUS * 2.0);
            reward += height_ratio.powi(2) * 0.25 * REWARD_MULTIPLIER;
        }

        // --- ConserveBoostReward × 0.8 * M / mtt ---
        // Reward having boost available (sqrt of fraction).
        {
            reward += boost_frac.sqrt() * 0.8 * REWARD_MULTIPLIER / MATCH_TIMEOUT_TICKS;
        }

        // --- HitBallHardReward × 0.1 * M ---
        // Reward large ball velocity changes on touch (hard hits).
        if ball_touched {
            let vel_change = v3_len(v3_sub(ball_vel, pre_ball_vel)) / BALL_MAX_SPEED;
            reward += vel_change * 0.1 * REWARD_MULTIPLIER;
        }

        // --- AlignBallGoalReward(defense=1, offense=1) × 0.2 * M / mtt ---
        // Reward positioning where: (a) ball is between car and orange goal (offensive alignment),
        // and (b) car is between ball and blue goal (defensive alignment).
        {
            // defensive: (ball - car) aligned with (car - blue_goal_back)
            let def = v3_dot(
                v3_norm(v3_sub(ball_pos, car_pos)),
                v3_norm(v3_sub(car_pos, BLUE_GOAL_BACK)),
            );
            // offensive: (ball - car) aligned with (orange_goal_back - car)
            let off = v3_dot(
                v3_norm(v3_sub(ball_pos, car_pos)),
                v3_norm(v3_sub(ORANGE_GOAL_BACK, car_pos)),
            );
            reward += (def + off) * 0.2 * REWARD_MULTIPLIER / MATCH_TIMEOUT_TICKS;
        }

        // --- DribbleReward × 0.5 * M / mtt ---
        // Reward when ball is directly above car (dribble position).
        if ball_touched {
            let rel_x = ball_pos.0 - car_pos.0;
            let rel_y = ball_pos.1 - car_pos.1;
            let horiz_dist = (rel_x * rel_x + rel_y * rel_y).sqrt();
            if horiz_dist < BALL_RADIUS && ball_pos.2 > car_pos.2 {
                reward += 0.5 * REWARD_MULTIPLIER / MATCH_TIMEOUT_TICKS;
            }
        }

        // --- HitPostPenalty × 0.1 * M / mtt (penalty, applied negative) ---
        // Penalize ball being wide of the posts near the orange back wall (post hit).
        {
            if ball_pos.0.abs() > GOAL_WIDTH / 2.0
                && ball_pos.1 > BACK_WALL_Y - 2.0 * BALL_RADIUS
            {
                reward -= 0.1 * REWARD_MULTIPLIER / MATCH_TIMEOUT_TICKS;
            }
        }

        // --- FirstTouchReward × 0.3 * M ---
        // One-time reward for making first ball contact after a kickoff.
        if !self.first_touch_given && ball_touched {
            reward += 0.3 * REWARD_MULTIPLIER;
        }

        // --- MaximizeTimeBetweenFlipsReward × 0.2 * M / mtt ---
        // Reward keeping forward vector pointed upward while a flip is available
        // (has_jumped && !has_double_jumped && !has_flipped => flip still available).
        if new_car.has_jumped && !new_car.has_double_jumped && !new_car.has_flipped {
            // forward.dot((0, 0, 1)) is just the Z component of the forward vector
            reward += car_fwd.2 * 0.2 * REWARD_MULTIPLIER / MATCH_TIMEOUT_TICKS;
        }

        // --- VelocityReward × 0.35 * M / mtt ---
        // Reward car speed as a fraction of max speed.
        {
            let speed = v3_len(car_vel) / CAR_MAX_SPEED;
            reward += speed * 0.35 * REWARD_MULTIPLIER / MATCH_TIMEOUT_TICKS;
        }

        // Restore original state
        let _ = arena.pin_mut().set_game_state(&original_state);

        reward as f64
    }

    fn apply_action(&mut self, action_idx: usize) -> Result<(), Box<dyn Error>> {
        let controls = self.lookup_table[action_idx];
        let mut arena = self.arena.borrow_mut();

        arena.pin_mut().set_car_controls(self.car_id, controls)?;
        arena.pin_mut().step(self.tick_skip as u32);

        // Track first ball touch for FirstTouchReward
        if !self.first_touch_given {
            let car = arena.pin_mut().get_car(self.car_id);
            let ball = arena.pin_mut().get_ball();
            let dx = car.pos.x - ball.pos.x;
            let dy = car.pos.y - ball.pos.y;
            let dz = car.pos.z - ball.pos.z;
            if (dx * dx + dy * dy + dz * dz).sqrt() < TOUCH_DIST_THRESHOLD {
                self.first_touch_given = true;
            }
        }

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
        self.first_touch_given = false;
        self.send_state_to_rlviser();
        Ok(())
    }
}

impl Drop for RocketSimEnvironment {
    fn drop(&mut self) {
        let current_vis = VISUALIZER_ID.load(std::sync::atomic::Ordering::Relaxed);
        if current_vis == self.id {
            VISUALIZER_ID.store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

// --- Vec3 helpers (tuples of f32) ---

fn v3_dot(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

fn v3_sub(a: (f32, f32, f32), b: (f32, f32, f32)) -> (f32, f32, f32) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

fn v3_scale(a: (f32, f32, f32), s: f32) -> (f32, f32, f32) {
    (a.0 * s, a.1 * s, a.2 * s)
}

fn v3_len(a: (f32, f32, f32)) -> f32 {
    (a.0 * a.0 + a.1 * a.1 + a.2 * a.2).sqrt()
}

fn v3_norm(a: (f32, f32, f32)) -> (f32, f32, f32) {
    let len = v3_len(a);
    if len < 1e-8 {
        (0.0, 0.0, 0.0)
    } else {
        v3_scale(a, 1.0 / len)
    }
}

// --- RLViser serialization (unchanged) ---

fn to_flat_vec3(v: rocketsim_rs::math::Vec3) -> flat_rs::Vec3 {
    flat_rs::Vec3 {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}

fn to_flat_mat3(m: rocketsim_rs::math::RotMat) -> flat_rs::Mat3 {
    flat_rs::Mat3 {
        forward: to_flat_vec3(m.forward),
        right: to_flat_vec3(m.right),
        up: to_flat_vec3(m.up),
    }
}

fn to_flat_game_state(state: &rocketsim_rs::GameState) -> flat_rs::GameState {
    flat_rs::GameState {
        tick_rate: state.tick_rate,
        tick_count: state.tick_count,
        game_mode: match state.game_mode {
            GameMode::Soccar => flat_rs::GameMode::Soccar,
            GameMode::Hoops => flat_rs::GameMode::Hoops,
            GameMode::Heatseeker => flat_rs::GameMode::Heatseeker,
            GameMode::Snowday => flat_rs::GameMode::Snowday,
            GameMode::Dropshot => flat_rs::GameMode::Dropshot,
            _ => flat_rs::GameMode::Soccar,
        },
        cars: Some(
            state
                .cars
                .iter()
                .map(|c| flat_rs::CarInfo {
                    id: c.id as u64,
                    team: match c.team {
                        Team::Blue => flat_rs::Team::Blue,
                        Team::Orange => flat_rs::Team::Orange,
                    },
                    state: Box::new(flat_rs::CarState {
                        physics: flat_rs::PhysState {
                            pos: to_flat_vec3(c.state.pos),
                            rot_mat: to_flat_mat3(c.state.rot_mat),
                            vel: to_flat_vec3(c.state.vel),
                            ang_vel: to_flat_vec3(c.state.ang_vel),
                        },
                        is_on_ground: c.state.is_on_ground,
                        wheels_with_contact: flat_rs::WheelsWithContact {
                            front_left: c.state.wheels_with_contact[0],
                            front_right: c.state.wheels_with_contact[1],
                            rear_left: c.state.wheels_with_contact[2],
                            rear_right: c.state.wheels_with_contact[3],
                        },
                        has_jumped: c.state.has_jumped,
                        has_double_jumped: c.state.has_double_jumped,
                        has_flipped: c.state.has_flipped,
                        flip_rel_torque: to_flat_vec3(c.state.flip_rel_torque),
                        jump_time: c.state.jump_time,
                        flip_time: c.state.flip_time,
                        is_flipping: c.state.is_flipping,
                        is_jumping: c.state.is_jumping,
                        air_time: c.state.air_time,
                        air_time_since_jump: c.state.air_time_since_jump,
                        boost: c.state.boost,
                        time_since_boosted: c.state.time_since_boosted,
                        is_boosting: c.state.is_boosting,
                        boosting_time: c.state.boosting_time,
                        is_supersonic: c.state.is_supersonic,
                        supersonic_time: c.state.supersonic_time,
                        handbrake_val: c.state.handbrake_val,
                        is_auto_flipping: c.state.is_auto_flipping,
                        auto_flip_timer: c.state.auto_flip_timer,
                        auto_flip_torque_scale: c.state.auto_flip_torque_scale,
                        world_contact_normal: None,
                        car_contact: None,
                        is_demoed: c.state.is_demoed,
                        demo_respawn_timer: c.state.demo_respawn_timer,
                        ball_hit_info: None,
                        last_controls: flat_rs::CarControls {
                            throttle: c.state.last_controls.throttle,
                            steer: c.state.last_controls.steer,
                            pitch: c.state.last_controls.pitch,
                            yaw: c.state.last_controls.yaw,
                            roll: c.state.last_controls.roll,
                            jump: c.state.last_controls.jump,
                            boost: c.state.last_controls.boost,
                            handbrake: c.state.last_controls.handbrake,
                        },
                    }),
                    config: flat_rs::CarConfig {
                        hitbox_size: to_flat_vec3(c.config.hitbox_size),
                        hitbox_pos_offset: to_flat_vec3(c.config.hitbox_pos_offset),
                        front_wheels: flat_rs::WheelPairConfig {
                            wheel_radius: c.config.front_wheels.wheel_radius,
                            suspension_rest_length: c.config.front_wheels.suspension_rest_length,
                            connection_point_offset: to_flat_vec3(
                                c.config.front_wheels.connection_point_offset,
                            ),
                        },
                        back_wheels: flat_rs::WheelPairConfig {
                            wheel_radius: c.config.back_wheels.wheel_radius,
                            suspension_rest_length: c.config.back_wheels.suspension_rest_length,
                            connection_point_offset: to_flat_vec3(
                                c.config.back_wheels.connection_point_offset,
                            ),
                        },
                        three_wheels: c.config.three_wheels,
                        dodge_deadzone: c.config.dodge_deadzone,
                    },
                })
                .collect(),
        ),
        ball: flat_rs::BallState {
            physics: flat_rs::PhysState {
                pos: to_flat_vec3(state.ball.pos),
                rot_mat: to_flat_mat3(state.ball.rot_mat),
                vel: to_flat_vec3(state.ball.vel),
                ang_vel: to_flat_vec3(state.ball.ang_vel),
            },
            hs_info: flat_rs::HeatseekerInfo {
                y_target_dir: state.ball.hs_info.y_target_dir,
                cur_target_speed: state.ball.hs_info.cur_target_speed,
                time_since_hit: state.ball.hs_info.time_since_hit,
            },
            ds_info: flat_rs::DropshotInfo {
                charge_level: state.ball.ds_info.charge_level,
                accumulated_hit_force: state.ball.ds_info.accumulated_hit_force,
                y_target_dir: state.ball.ds_info.y_target_dir,
                has_damaged: state.ball.ds_info.has_damaged,
                last_damage_tick: state.ball.ds_info.last_damage_tick,
            },
        },
        pads: None,
        tiles: None,
    }
}
