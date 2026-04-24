# CSDP-RS

An experimental research playground for biologically-motivated reinforcement learning algorithms that do not use backpropagation. The framework implements and compares two families of local learning models -- Forward-Forward networks and Contrastive Signal Dependent Plasticity spiking networks -- across a variety of RL algorithm designs, from simple contrastive action evaluators up through actor-critic and PPO-style approaches.

The goal is to explore whether locally-computed credit signals (goodness, spike activity, reward modulation) can substitute for gradient-based backprop in online RL settings.

---

## Models

### Forward-Forward (FF)

Introduced by Hinton (2022), the Forward-Forward algorithm replaces the forward-backward pass of backprop with two forward passes: one on positive (real/good) data and one on negative (bad/synthetic) data. Each layer learns independently by maximizing a scalar "goodness" (sum of squared activations) on positive data and minimizing it on negative data. No gradients flow between layers.

Two FF model variants are used in this framework:

- **FFModel** -- a standard feedforward network where each layer trains independently via contrastive goodness. Input is a concatenation of state and action (or state + action + goal). Used by algorithms FF1 through FFSAC.

- **FFMultiModel** -- a multi-output variant where each layer produces `num_classes` parallel channels. Used for return distribution prediction, where the model learns to classify discretized Monte Carlo returns. Each layer trains via either contrastive binary labels or supervised class labels. Used by FFMulti1, FFMulti2, and FF_PPO.

### CSDP (Contrastive Signal Dependent Plasticity)

CSDP is a biologically-motivated learning rule for spiking neural networks (SNNs). Weights are updated locally using a contrastive Hebbian-style rule modulated by a reward or label signal. The positive phase (label=1) strengthens synaptic weights when both pre- and post-synaptic neurons fire together; the negative phase (label=0) weakens them. There is no global loss function and no backpropagation.

Neurons use the Leaky Integrate-and-Fire (LIF) model. Input layers use Bernoulli stochastic spike encoding. Each synapse stores forward and backward weight matrices. A context signal (the contrastive label) gates which update direction is applied.

Four CSDP model variants are used:

- **RLModel1** -- evaluates a single (state, action_index) pair per forward pass. Used by CSDP1.
- **RLModel2** -- evaluates a (state_t, state_{t+M}, action) triple; used for transition-based contrastive learning. Used by CSDP2 and CSDP4.
- **RLModel3** -- a dual actor-critic architecture with separate SNN modules for policy and value estimation. Used by CSDP3.
- **CSDPMultiModel** -- a multi-class SNN where hidden layer neurons are divided into `num_classes` groups, each representing a discretized return bin. Used by CSDP5.

---

## Algorithms

Algorithms are ordered by number, with PPO last. All algorithms implement the `Algorithm` trait and are selected via the `--algo` flag.

---

### CSDP1 -- Basic SNN Action Evaluator

**Model**: RLModel1 (CSDP, LIF neurons)  
**Network input**: `[state | action_index]` evaluated independently per action  
**Network output**: Scalar goodness (accumulated spike activity over 40 timesteps)  
**Environments**: 1  
**Episodes**: 100, Steps/episode: 50, Epochs/episode: 5

The simplest CSDP algorithm. The SNN is queried once per action at each step; the action with the highest spike activity is selected. After each episode, training pairs are formed by ranking actions by reward and assigning contrastive labels proportional to their rank. The model learns to associate high-activity states with high-reward actions via the CSDP weight update.

```
for each episode:
    disable_learning()
    for each step:
        for each action a:
            activity[a] = snn_forward(state, a, T=40 timesteps)
        best_action = argmax(activity)
        apply_action(best_action)
        record (state, action, reward) for all actions

    enable_learning()
    sort episode transitions by reward
    assign label[i] = rank[i] / total_transitions  (normalized rank)
    for each epoch:
        for each transition:
            snn_train(state, action, label=normalized_rank)
```

---

### CSDP2 -- Transition Predictor

**Model**: RLModel2 (CSDP, LIF neurons)  
**Network input**: `[state_t | state_{t+1} | action]` -- evaluates whether an action caused an observed state transition  
**Network output**: Scalar prediction score  
**Environments**: 1  
**Episodes**: 100, Steps/episode: 50, Epochs/episode: 5

Rather than directly predicting reward, CSDP2 learns a contrastive transition model. Positive samples are actual (s_t, s_{t+1}, a_t) triplets from the trajectory. Negative samples are formed by substituting a random alternative action, breaking the causal link. Rewards modulate the strength of the update via sigmoid scaling on the output layer.

```
for each episode:
    collect trajectory: [(state_t, action_t, reward_t, state_{t+1}), ...]

    for each transition (s_t, a_t, r_t, s_{t+1}):
        # positive: real (state, next_state, action) triplet
        pos_input = [s_t | s_{t+1} | a_t]
        reward_scale = sigmoid(r_t)  # modulates learning strength

        # negative: same (s_t, s_{t+1}) but with a random action
        neg_input = [s_t | s_{t+1} | random_action]

        snn_train(pos_input, label=1.0, reward_scale=reward_scale)
        snn_train(neg_input, label=0.0)
```

---

### CSDP3 -- Actor-Critic with Contrastive Learning

**Model**: RLModel3 (CSDP, dual actor + critic SNNs)  
**Network input (actor)**: `[state]` -- outputs a spike-rate action vector  
**Network input (critic)**: `[state | action_z]` -- evaluates Q(s, a)  
**Network output (actor)**: Continuous action via accumulated spike rates  
**Network output (critic)**: Scalar Q-value estimate  
**Environments**: 1  
**Episodes**: 100, Steps/episode: 50

A CSDP actor-critic. The actor generates actions by accumulating LIF spike rates over multiple timesteps into a continuous action vector. A noise-perturbed baseline action is used for exploration. The critic evaluates both the actor's output and the baseline contrastively using TD targets. Actor weights are updated by labeling the higher-Q action as positive and the lower-Q action as negative.

```
for each episode:
    for each step:
        action_z = actor_snn_forward(state, T timesteps)  # spike-rate action
        baseline_z = action_z + noise

        Q_action   = critic_snn_forward(state, action_z)
        Q_baseline = critic_snn_forward(state, baseline_z)

        # Actor update: whichever action had higher Q is the positive sample
        if Q_action > Q_baseline:
            actor_train(state, action_z,   label=1.0)
            actor_train(state, baseline_z, label=0.0)
        else:
            actor_train(state, baseline_z, label=1.0)
            actor_train(state, action_z,   label=0.0)

        # Critic TD update
        r = env.step(action_z)
        next_Q = critic_snn_forward(next_state, action_z)
        td_target = r + gamma * next_Q
        critic_train(state, action_z, label=(td_target > Q_baseline))
```

---

### CSDP4 -- Vectorized Replay-Buffer RL

**Model**: RLModel2 (CSDP, LIF neurons)  
**Network input**: `[state_t | state_{t+1} | action]`  
**Network output**: Scalar score  
**Environments**: 16 parallel  
**Episodes**: 500, Steps/episode: 70

CSDP4 adds a replay buffer and vectorized environments to CSDP2's transition-prediction approach. Actions are selected via softmax over per-action scores (temperature tau=0.07). Discounted Monte Carlo returns are computed at the end of each episode. Training samples the top-256 highest-return transitions as positives and pairs them with randomly chosen alternative actions as negatives. Updates are batched rather than per-sample.

```
replay_buffer = []  # max 200k transitions

for each episode:
    run 16 envs in parallel for N steps
    compute discounted returns G_t for each trajectory
    store (state_t, action_t, G_t, state_{t+1}) in replay_buffer

    # action selection at inference:
    for each action a:
        score[a] = snn_forward([state | state_prev | a])
    action = sample from softmax(score, tau=0.07)

    # training:
    sort replay_buffer by G descending
    positives = top 256 transitions by G
    for each positive (s_t, a_t, G_t, s_{t+1}):
        neg_action = random_action != a_t
        pos_input = [s_t | s_{t+1} | a_t]
        neg_input = [s_t | s_{t+1} | neg_action]
        snn_train_batch(pos_input, label=1.0)
        snn_train_batch(neg_input, label=0.0)
```

---

### CSDP5 -- Multi-Class Monte Carlo SNN

**Model**: CSDPMultiModel (CSDP, LIF, multi-class output groups)  
**Network input**: `[state_t | state_diff | one_hot(action) * 3.0]`  
**Network output**: Softmax distribution over 50 return classes  
**Environments**: 16 parallel  
**Episodes**: 500, Steps/episode: 70

CSDP5 discretizes Monte Carlo returns into 50 bins and trains the SNN to predict which return bin corresponds to a given (state, action) pair. Hidden layer neurons are partitioned into 50 groups, one per class. Action selection computes an expected return as a weighted sum of class centers, then applies epsilon-greedy selection. Class boundaries are adapted via exponential moving average of observed return percentiles. Checkpoints are saved every 50 episodes and training can be resumed with `--resume`.

```
n_classes = 50
class_bounds = initialize_percentile_bounds()
epsilon = 1.0  # decays over training

replay_buffer = []

for each episode:
    run 16 envs for N steps with epsilon-greedy action selection:
        for each action a:
            class_logits = csdp_multi_forward([state | state_diff | one_hot(a)])
            probs = softmax(class_logits)
            expected_return[a] = sum(probs * class_centers)
        action = epsilon_greedy(expected_return)

    compute MC returns G_t (gamma=0.9)
    store transitions in replay_buffer

    # update class boundaries via EMA
    class_bounds = EMA(class_bounds, percentiles(all_G_t), alpha=0.1)

    # training
    sample batch from replay_buffer
    for each (state, action, G_t):
        target_class = digitize(G_t, class_bounds)
        csdp_train([state | state_diff | one_hot(action)], label=target_class)
```

---

### CSDP_PPO -- PPO with CSDP Spiking Q-Function

**Model**: CSDPMultiModel (CSDP, LIF, multi-class output groups)  
**Network input**: `[normalized_state | one_hot(action) * 3.0]`  
**Network output**: Softmax distribution over 50 return classes  
**Environments**: 8 parallel  
**Episodes**: 1000, Steps/episode: 64, PPO epochs/rollout: 2

CSDP_PPO applies the PPO training loop to a CSDP spiking network used as a distributional Q-function. A single `CSDPMultiModel` maps (state, action) pairs to a distribution over 50 discretized return bins, giving Q(s, a) ≈ E[G | s, a]. Action selection is argmax over E[Q(s, a)] — the expected return under the class distribution — with an epsilon-greedy warmup that decays to zero, producing committed non-jittery behavior. Generalized Advantage Estimation (GAE, lambda=0.95, gamma=0.99) computes return targets. Each transition's GAE return target is converted to a class label and used to train the Q-function via CSDP's contrastive STDP rule. Unlike CSDP5, there is no replay buffer: data is strictly on-policy and replaced each episode. Return class boundaries adapt via EMA over observed return percentiles. Checkpoints save to `checkpoints/csdp_ppo/` and training can be resumed with `--resume`.

```
n_classes = 50
class_bounds = initial_percentile_bounds()
epsilon = 0.3  # decays to 0 over 200 episodes

model = CSDPMultiModel(input=state_size + action_size, classes=50)

for each episode:
    collect rollout: run 8 envs for 64 steps each
        for each env, action selection:
            for each action a:
                class_logits = model_forward([state | one_hot(a) * 3.0])
                expected_return[a] = sum(softmax(class_logits) * class_centers)
            action = epsilon_greedy(argmax(expected_return))

    bootstrap V(s_T) = max_a E[Q(s_T, a)] for each env

    compute GAE advantages A_t and return targets R_t (gamma=0.99, lambda=0.95):
        delta_t = r_t + gamma * V_{t+1} - V_t
        A_t = sum_{k>=t} (gamma * lambda)^{k-t} * delta_k
        R_t = A_t + V_t

    normalize advantages (mean=0, std=1)
    update class_bounds via EMA on observed returns R_t

    for each PPO epoch (2 epochs):
        shuffle transitions
        for each (state, action, R_t):
            target_class = digitize(R_t, class_bounds)
            model_train([state | one_hot(action) * 3.0], label=target_class)
```

---

### FF1 -- State/Action Iterator

**Model**: FFModel (Forward-Forward)  
**Network input**: `[one_hot(action) | state]`  
**Network output**: Scalar goodness per action  
**Environments**: 200 parallel  
**Episodes**: 100, Steps/episode: 50, Epochs/episode: 100

The baseline Forward-Forward RL algorithm. 200 environments run in parallel. At each step, all actions are evaluated in a single batched forward pass. The best and worst actions by reward are selected as the positive and negative samples for FF training. The network learns to output high goodness for rewarding actions and low goodness for poor ones.

```
envs = 200 parallel environments

for each episode:
    reset all envs
    for each step:
        for each env e, for each action a:
            input[e,a] = [one_hot(a) | state_e]
        goodness[e,a] = ff_forward(input[e,a])
        best_action[e] = argmax(goodness[e,:])
        apply best_action[e] to env e

        for each env e:
            sort actions by reward
            pos_input = input[e, best_action]
            neg_input = input[e, worst_action]
            ff_train(pos_input, neg_input)  # 100 local epochs
```

---

### FF2 -- Transition Evaluator

**Model**: FFModel (Forward-Forward)  
**Network input**: `[action | state_t | goal_state]` where goal_state = state space upper bounds  
**Network output**: Scalar goodness  
**Environments**: 16 parallel  
**Episodes**: 100, Steps/episode: 50

FF2 extends FF1 by including a goal state in the input, training the network to evaluate transitions toward that goal. Positive samples are (action, s_t, s_{t+1}) triplets where the action was actually taken. Negative samples use the same (s_t, s_{t+1}) but with a random alternative action, teaching the model that the actual action caused the observed transition.

```
goal_state = env.state_bounds()  # upper bound of each state dimension

for each episode:
    run 16 envs for N steps, collecting (s_t, a_t, s_{t+1})

    for each transition:
        pos_input = [a_t | s_t | goal_state]
        neg_action = random_action != a_t
        neg_input  = [neg_action | s_t | goal_state]
        ff_train(pos_input, neg_input)
```

---

### FF3 -- Probabilistic Rank Trajectory

**Model**: FFModel (Forward-Forward)  
**Network input**: `[one_hot(action) | state]`  
**Network output**: Scalar goodness  
**Environments**: 16 parallel  
**Episodes**: 100, Steps/episode: 50

FF3 assigns probabilistic positive/negative labels based on trajectory rank rather than per-step action quality. At the end of each episode, environments are ranked by total reward. Each trajectory is labeled with a positive probability proportional to its rank (p_pos = rank / n_envs). Trajectories are then sampled stochastically as positive or negative training data.

```
for each episode:
    run 16 envs for N steps, record total_reward[e] for each env e
    sort envs by total_reward ascending
    assign p_pos[e] = rank[e] / n_envs

    for each transition (s_t, a_t) from env e:
        if random() < p_pos[e]:
            ff_train(input, positive=True)
        else:
            ff_train(input, positive=False)
```

---

### FF4 -- Temporal Contrastive RL

**Model**: FFModel (Forward-Forward)  
**Network input**: `[one_hot(action) | state]`  
**Network output**: Scalar goodness  
**Environments**: 16 parallel  
**Episodes**: 500, Steps/episode: 70

FF4 introduces a replay buffer and discounted return-based contrastive training, mirroring the design of CSDP4 but with a FF network. Actions are selected via Boltzmann softmax over goodness scores. After each episode, the buffer is sorted by return and the top-256 transitions form the positive batch, while randomly sampled alternative actions form the negative batch.

```
replay_buffer = []  # max 200k transitions

for each episode:
    run 16 envs for N steps
    compute discounted returns G_t (gamma=0.99)
    add (state, action, G_t) to replay_buffer

    # action selection:
    for each action a:
        goodness[a] = ff_forward([one_hot(a) | state])
    action = sample from softmax(goodness, tau=0.07)

    # training:
    sort replay_buffer by G_t descending
    positives = top 256 by G_t
    for each (state, action, G_t) in positives:
        neg_action = random_action != action
        pos_input = [one_hot(action)     | state]
        neg_input = [one_hot(neg_action) | state]
        ff_train(pos_input, neg_input)
```

---

### FFSAC -- Forward-Forward Soft Actor-Critic

**Model**: FFModel (Forward-Forward)  
**Network input**: `[one_hot(action) | state]`  
**Network output**: Scalar goodness (treated as Q-value)  
**Environments**: 16 parallel  
**Episodes**: 500, Steps/episode: 70

FFSAC adapts the Soft Actor-Critic formulation to FF networks. Goodness scores are treated as Q-values. The soft value function is computed as V(s) = tau * log(sum_a exp(Q(s,a) / tau)), and TD targets are formed as r + gamma * V(s'). Actions are selected by Boltzmann softmax. Training uses a delta-based rule: if the TD delta is positive, the actual action is the positive sample; if negative, a random action is used instead.

```
replay_buffer = []  # max 50k transitions
tau_sac = 0.1

for each episode:
    run 16 envs, selecting actions via Boltzmann softmax over goodness
    store (state, action, reward, next_state) in replay_buffer

    for each sampled batch:
        # soft value of next state
        for each action a:
            Q_next[a] = ff_forward([one_hot(a) | next_state])
        V_next = tau_sac * log(sum_a exp(Q_next[a] / tau_sac))

        td_target = reward + gamma * V_next

        for each action a:
            Q_current[a] = ff_forward([one_hot(a) | state])
        V_current = tau_sac * log(sum_a exp(Q_current[a] / tau_sac))

        delta = td_target - V_current

        if delta > 0.1:
            pos_input = [one_hot(actual_action) | state]
            neg_input = [one_hot(random_action) | state]
        elif delta < -0.1:
            pos_input = [one_hot(random_action) | state]
            neg_input = [one_hot(actual_action) | state]
        ff_train(pos_input, neg_input)
```

---

### FFMulti1 -- Multi-Class Classification RL

**Model**: FFMultiModel (Forward-Forward, multi-class output)  
**Network input**: `[one_hot(action) | state]`  
**Network output**: Per-class goodness scores  
**Environments**: 16 parallel  
**Episodes**: 500, Steps/episode: 70

FFMulti1 uses the multi-output FF model with a replay buffer. The network learns to classify state-action pairs according to their return rank. The highest-return transitions are labeled as class 1 (positive), and low-return transitions are labeled as class 0 (negative). A single batched forward pass over all actions and states evaluates all options simultaneously.

```
replay_buffer = []  # max 200k transitions

for each episode:
    batch infer over all (env, action) pairs simultaneously
    select best action per env from class scores
    apply actions, store transitions with discounted returns

    sort replay_buffer by G descending
    positives = top 256 transitions
    negatives = bottom 256 transitions

    ff_multi_train(positives, class_label=1)
    ff_multi_train(negatives, class_label=0)
```

---

### FFMulti2 -- Return Distribution Prediction

**Model**: FFMultiModel with 50 return classes; separate main and target models  
**Network input**: `[one_hot(action) * 3.0 | state]` (state normalization varies by environment)  
**Network output**: Softmax distribution over 50 return classes  
**Environments**: 16 parallel  
**Episodes**: 100, Steps/episode: 500, Epochs/episode: 12

The most developed FF algorithm. FFMulti2 discretizes MC returns into 50 bins and trains the multi-class FF model to predict the return class for a given (state, action) pair, analogous to distributional RL. It maintains a main model and a target model (hard-synced every 10 episodes). Class boundaries are adapted online via EMA over observed return percentiles. Epsilon-greedy exploration decays over the first 200 episodes. Training uses 1024-sample batches. Checkpoints save both model weights and training metadata (class bounds, reward history, episode counter) to `checkpoints/ff_multi2/`.

```
n_classes = 50
class_bounds = initial_percentile_bounds()
epsilon = 1.0  # decays to 0 over 200 episodes
target_model = copy(main_model)

for each episode:
    if episode % 10 == 0:
        target_model = copy(main_model)  # hard sync

    run 16 envs for 500 steps with epsilon-greedy:
        for each action a:
            logits = main_model_forward([one_hot(a) * 3.0 | state])
            probs = softmax(logits)
            expected_return[a] = sum(probs * class_centers)
        action = epsilon_greedy(expected_return)

    compute MC returns G_t (gamma=0.99)
    store (state, action, G_t) transitions

    # update class bounds via EMA
    class_bounds = EMA(class_bounds, percentiles(all_G_t), alpha=0.1)

    for each training epoch (12 epochs):
        batch = sample 1024 transitions
        for each (state, action, G_t) in batch:
            target_class = digitize(G_t, class_bounds)
            ff_multi_train([one_hot(action) * 3.0 | state], label=target_class)
```

---

### FF_PPO -- PPO with Forward-Forward Models

**Model**: Two FFMultiModel instances -- one policy model and one value model (50 return classes each)  
**Network input**: `[one_hot(action) * 3.0 | state]`  
**Network output (policy)**: Class distribution used to rank action preferences  
**Network output (value)**: Expected return estimate (used as TD baseline)  
**Environments**: 16 parallel  
**Episodes**: 100, Steps/episode: 512, PPO epochs/rollout: 2

The most complex algorithm in the framework. FF_PPO adapts Proximal Policy Optimization to local FF learning. A separate policy model and value model are trained using goodness signals rather than gradients. Generalized Advantage Estimation (GAE, lambda=0.95, gamma=0.99) is used to compute advantages. Rather than gradient-based clipping, the advantage sign gates the contrastive label: positive advantages cause the taken action to be the positive training sample; negative advantages flip the labels. Entropy is encouraged with a small coefficient (0.001). Both models are checkpointed to `checkpoints/ff_ppo/`.

```
n_classes = 50
class_bounds = initial_percentile_bounds()
epsilon = 0.3  # epsilon-greedy warmup, decays to 0 over 200 episodes

policy_model = FFMultiModel(n_classes=50)
value_model  = FFMultiModel(n_classes=50)

for each episode:
    collect rollout: run 16 envs for 512 steps each

    compute MC returns G_t (gamma=0.99)
    compute value estimates V_t from value_model expected return
    compute GAE advantages A_t (lambda=0.95):
        delta_t = r_t + gamma * V_{t+1} - V_t
        A_t = sum_{k>=t} (gamma * lambda)^{k-t} * delta_k

    update class_bounds via EMA on observed returns

    for each PPO epoch (2 epochs):
        for each transition (state, action, A_t, G_t):
            # policy update gated by advantage sign
            if A_t > 0:
                pos = [one_hot(action)        * 3.0 | state]
                neg = [one_hot(random_action) * 3.0 | state]
            else:
                pos = [one_hot(random_action) * 3.0 | state]
                neg = [one_hot(action)        * 3.0 | state]
            policy_train(pos, neg)

            # value update: predict return class
            target_class = digitize(G_t, class_bounds)
            value_train([one_hot(action) * 3.0 | state], label=target_class)
```

---

## CLI Arguments

The main binary accepts the following arguments:

```
cargo run --release [-- [OPTIONS]]
```

| Argument | Description |
|---|---|
| `--algo <name>` | Algorithm to run (default: `csdp2`). See table below. |
| `--grid` | Use the Grid environment (discrete state space). |
| `--rocketsim` | Use the RocketSim environment (Rocket League simulator). |
| `--visualize` / `-v` | Enable the Ratatui TUI with live training graphs and layer activity. Spike history panels are only populated for CSDP algorithms. |
| `--infinite-epochs` | Run until interrupted (sets episode count to `usize::MAX`). |
| `--resume` | Load from checkpoint and resume training. Supported by `ff_multi2`, `ff_ppo`, `csdp5`, and `csdp_ppo`. |

If no environment flag is given, the binary attempts to connect to a physical LeRobot arm over serial. If that connection fails, it falls back to the Grid environment automatically.

**Algorithm names for `--algo`:**

| Name | Algorithm |
|---|---|
| `csdp1` | Basic SNN action evaluator |
| `csdp2` | SNN transition predictor (default) |
| `csdp3` | SNN actor-critic |
| `csdp4` | Vectorized SNN with replay buffer |
| `csdp5` | Multi-class Monte Carlo SNN |
| `csdp_ppo` | PPO with CSDP spiking Q-function |
| `ff1` | FF state/action evaluator |
| `ff2` | FF transition evaluator with goal state |
| `ff3` | FF probabilistic rank trajectory |
| `ff4` | FF temporal contrastive RL with replay buffer |
| `ffsac` | FF soft actor-critic |
| `ff_multi1` | Multi-class FF with binary class labels |
| `ff_multi2` | Multi-class FF return distribution prediction |
| `ff_ppo` | PPO-style training with dual FF models |

**Examples:**

```bash
# Run FFMulti2 on the grid environment with visualization
cargo run --release -- --algo ff_multi2 --grid --visualize

# Resume a previous ff_ppo checkpoint
cargo run --release -- --algo ff_ppo --grid --resume

# Run CSDP5 indefinitely on RocketSim
cargo run --release -- --algo csdp5 --rocketsim --infinite-epochs
```

---

## Additional Binaries

These tools are built separately from the main training binary and support the physical robot workflow.

| Binary | Command | Description |
|---|---|---|
| `collect_data` | `cargo run --bin collect_data` | Records joint positions from a physical LeRobot arm to a timestamped CSV file over serial. Used to collect demonstration data. |
| `playback_data` | `cargo run --bin playback_data -- <file.csv>` | Replays a recorded CSV trajectory on the physical robot, resampling to match the original timing. |
| `teleoperate` | `cargo run --bin teleoperate` | Leader-follower teleoperation: streams joint positions from a leader robot to a follower robot in real time over two serial connections. |
| `test_mnist_ff` | `cargo run --bin test_mnist_ff` | Downloads MNIST and trains an FFMultiModel on digit classification. Used to validate the FF multi-class model outside of RL. |
| `test_distributional` | `cargo run --bin test_distributional` | Interactive TUI for testing the distributional value head. Visualizes return class histograms for both FFMultiModel and CSDPMultiModel side by side. |

---

## Incomplete and Known Limitations

**PPO clipping not implemented.** `ff_ppo` uses the PPO name and GAE but does not implement the clipped surrogate objective. The policy update is binary contrastive (flip labels based on advantage sign), which is simpler and less principled than the standard PPO clipped ratio. The entropy coefficient (0.001) is also likely too small to meaningfully affect exploration.

**FFSAC has no target network.** The SAC implementation uses a single model for both current and target Q-value estimates. Standard SAC uses a slowly updated or hard-synced target network to stabilize TD targets. Here both action selection and TD target computation read from the same live model.

**CSDP3 critic instability.** The actor-critic CSDP variant had adaptive thresholding removed due to training instability. The critic baseline can drift during long runs, and comments in the code indicate this is an open issue without a clean fix yet.

**CSDP5 class bounds warm-start.** Early in training when the replay buffer has few samples, the EMA-updated class boundaries may collapse or become degenerate. There is no warm-start or fallback initialization for the bounds.

**FFMulti1 class labels are binary.** Despite using the multi-class output model, FFMulti1 only assigns class 0 or class 1 (worst vs. best transitions). The per-class output heads are not used to their full capacity in this algorithm.

**TUI visualization is CSDP-only.** The spike history panels and synapse weight visualizations in the Ratatui TUI are only wired up for CSDP models. FF algorithms return an error when the TUI tries to read a visualization snapshot, so the corresponding panels remain empty.

**Webcam integration is unused.** `nokhwa` is listed as a dependency for webcam capture, but no algorithm or tool currently uses it. It appears to be a placeholder for future vision-based state input.

**Vectorized algorithms cannot run on the physical robot.** Algorithms that spawn 16 parallel environments (FF4, FFMulti2, CSDP4, CSDP5, FF_PPO) rely on `clone_box()` to duplicate the environment, which is not supported by the physical robot interface. Running these without `--grid` or `--rocketsim` will either fall back to Grid or fail at environment construction.
