use crate::models::Model;
use candle_core::{Device, Result as CandleResult};

pub struct RLModel3 {
    pub actor: Model,
    pub critic: Model,
    pub device: Device,
    pub dt: f32,
}

impl RLModel3 {
    pub fn new(
        state_size: usize,
        action_size: usize,
        actor_hidden: Vec<usize>,
        critic_hidden: Vec<usize>,
        device: &Device,
        dt: f32,
        input_bounds: Option<Vec<usize>>,
    ) -> Option<Self> {
        let actor = Model::new(
            state_size,
            action_size,
            actor_hidden,
            device,
            dt,
            input_bounds,
        )?;

        // Critic takes state + action as input, outputs a single scalar value.
        // The context size for critic is 1 (the label).
        let critic = Model::new(state_size + action_size, 1, critic_hidden, device, dt, None)?;

        Some(Self {
            actor,
            critic,
            device: device.clone(),
            dt,
        })
    }

    pub fn enable_learning(&mut self) {
        self.actor.is_learning = true;
        self.critic.is_learning = true;
    }

    pub fn disable_learning(&mut self) {
        self.actor.is_learning = false;
        self.critic.is_learning = false;
    }

    pub fn reset_actor(&mut self, batch_size: usize) -> CandleResult<()> {
        self.actor.reset(batch_size)
    }

    pub fn reset_critic(&mut self, batch_size: usize) -> CandleResult<()> {
        self.critic.reset(batch_size)
    }
}
