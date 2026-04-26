# Presentation Outline: Reinforcement Learning Without Backpropagation

**Duration:** 20 Minutes
**Topic:** Exploring Contrastive Methods (CSDP and Forward-Forward) for RL

---

## 1. Introduction (3 Minutes)
- **The Problem with Backpropagation (BP):**
  - Biological implausibility (the "weight transport problem").
  - High memory requirements for storing intermediate activations.
  - Global synchronization requirements.
- **The Goal:** Finding local, biologically inspired learning rules that can still solve complex RL tasks.
- **Contrastive Learning:** Brief mention of the core idea — comparing "good" (positive) vs "bad" (negative) data.

## 2. Background: Contrastive Signal-Dependent Plasticity (CSDP) (4 Minutes)
- **What is CSDP?**
  - A local learning rule where weight updates depend on a global reinforcement signal and local activity.
  - No gradient flow backwards through layers.
- **Main Technique:**
  - Correlation between input, output, and a reward-modulated signal.
  - Weight update formula: $\Delta w = \eta \cdot (R - \bar{R}) \cdot e_{ij}$ where $e_{ij}$ is an eligibility trace.
- **Development Path (Algorithms 1-5):**
  - Iterative refinements handling stability and exploration.
  - **Deep Dive: CSDP5** — Implementation of a multi-class Monte Carlo Spiking Neural Network (SNN) that represents return distributions locally.
- **Data Highlight:** Stability achieved in multi-layer architectures without backprop.

## 3. Background: The Forward-Forward (FF) Algorithm (4 Minutes)
- **Geoffrey Hinton's FF Proposal:**
  - Replace forward and backward passes with two forward passes.
  - Positive Pass: Real data, increase "goodness."
  - Negative Pass: Generated/incorrect data, decrease "goodness."
- **Main Technique: FF-Multi2**
  - Multi-layer dense representation.
  - Classification-based RL: Discretizing the return space into "classes" to leverage contrastive classification loss.


## 4. Deep Dive: FF_PPO - The Final Frontier (6 Minutes)
- **Motivation:** Combining the stability of Proximal Policy Optimization (PPO) with the Forward-Forward backbone.
- **Architecture:**
  - Forward-Forward Multi-layer model as the backbone.
  - Separate Policy and Value representations, both trained without standard BP.
- **Technical Nuances:**
  - **Distributional Value:** Using classes to represent return ranges (`class_to_value`, `value_to_class`).
  - **Epsilon-Greedy & Temperature Schedules:** Balancing exploration in a non-gradient environment.
  - **Goodness Clamping:** Preventing saturation during training.
- **Advantages:**
  - Local updates.
  - Potential for massive parallelism.
  - Reduced memory footprint compared to storing full computational graphs.

## 5. Results & Discussion (2 Minutes)
- **Performance Comparison:**
  - Show plots of Reward vs. Epoch for CSDP, FF-Multi, and FF-PPO.
  - FF-PPO shows the most consistent convergence on complex environments.
- **Current Limitations:**
  - Scaling to very high-dimensional state spaces.
  - Tuning the "negative data" generation for RL (using off-policy or random actions).

## 6. Conclusion & Future Work (1 Minute)
- **Summary:** Contrastive methods are a viable alternative to BP for RL.
- **Future:** Spiking neural networks, real-robot implementation (Lerobot integration), and more sophisticated local rules.
- **Q&A**

---

# Data Collection for Plotting

| Algorithm | Key Stats to Plot | Source File |
|-----------|-------------------|-------------|
| **CSDP5** | Episode Rewards | `checkpoints/csdp5/training_state.json` |
| **FF-Multi2** | Epoch Rewards | `checkpoints/ff_multi2/epoch_rewards.csv` |
| **FF-PPO** | Epoch Rewards, Entropy, LR | `checkpoints/ff_ppo/epoch_rewards.csv` |

*(Note: Data has been extracted into the `collected_data/` directory for matplotlib processing.)*
