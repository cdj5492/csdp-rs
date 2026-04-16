pub mod reward_modulated;
pub mod standard;
pub mod multi_class;

use candle_core::{Result as CandleResult, Tensor};

/// Defines the generator for modulatory signals used by synapses.
pub trait ModSignalGenerator: Send + Sync {
    /// Calculate and update the internally stored modulatory signal.
    /// `spikes` is the current spiking activity.
    /// `label` is the target value.
    /// `reward` is the batched environment training scalar.
    /// `dt` is the timestep.
    fn calc_mod_signal(
        &mut self,
        spikes: &Tensor,
        label: &Tensor,
        reward: &Tensor,
        dt: f32,
    ) -> CandleResult<()>;

    /// Retrieve the calculated modulatory signal.
    fn get_mod_signal(&self) -> &Tensor;
}
