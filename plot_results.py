import csv
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return None
    epochs = []
    rewards = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            rewards.append(float(row['reward']))
    return {'epoch': epochs, 'reward': rewards}

def plot_all():
    ff4 = load_data('collected_data/ff4_rewards.csv')
    csdp4 = load_data('collected_data/csdp4_rewards.csv')
    csdp5 = load_data('collected_data/csdp5_rewards.csv')
    ff_multi = load_data('collected_data/ff_multi2_rewards.csv')
    ff_ppo = load_data('collected_data/ff_ppo_rewards.csv')

    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # 1. Plot CSDP4, CSDP5 and FF-Multi, FF4 on the left axis
    if ff4 is not None:
        ax1.plot(ff4['epoch'], ff4['reward'], label='FF4 (Temporal Contrastive)', color='purple', alpha=0.5)

    if csdp4 is not None:
        ax1.plot(csdp4['epoch'], csdp4['reward'], label='CSDP4', color='cyan', alpha=0.4, linestyle=':')
    
    if csdp5 is not None:
        ax1.plot(csdp5['epoch'], csdp5['reward'], label='CSDP5 (SNN)', color='blue', alpha=0.5)
    
    if ff_multi is not None:
        ax1.plot(ff_multi['epoch'], ff_multi['reward'], label='FF-Multi2 (Dense)', color='orange', alpha=0.5)
        
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward (CSDP5 / FF-Multi2)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')

    # 2. Plot FF-PPO on the right axis due to different scale (averaged reward)
    if ff_ppo is not None:
        ax2 = ax1.twinx()
        ax2.plot(ff_ppo['epoch'], ff_ppo['reward'], label='FF-PPO (Policy Gradient)', color='green', linewidth=2)
        ax2.set_ylabel('FF-PPO Reward', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.legend(loc='upper right')

    plt.title('Performance Comparison: Contrastive RL Algorithms (500 Episodes)')
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_performance.png', dpi=300)
    print("Plot saved to training_performance.png")

if __name__ == "__main__":
    plot_all()
