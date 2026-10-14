import numpy as np
import pandas as pd
import joblib
import os
import glob
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from collections import Counter  # Used for step distribution statistics

# ------------------------------
# 1. Load Random Forest Surrogate Model
# ------------------------------
try:
    optimized_rf_model = joblib.load(
        r'surrogate_model\random_forest_best_model.pkl'
    )
    print("Random Forest model loaded successfully!")
except FileNotFoundError:
    print("Error: Random Forest model file not found. Please verify the file path.")
    exit()

# ------------------------------
# 2. Input Action Space Range from User
# ------------------------------
try:
    action_space_input = input("Enter action space range [lower, upper, step] (e.g., [-0.2,0.2,0.01]): ")
    action_space_values = [float(x) for x in action_space_input.strip('[]').split(',')]
    lower_bound, upper_bound, step_size = action_space_values
except ValueError:
    print("Error: Please enter valid values in the format [lower, upper, step]!")
    exit()

# ------------------------------
# 3. Environment Definition (SimulationEnv)
# ------------------------------
class SimulationEnv:
    def __init__(self, initial_values, stop_threshold=2, use_visited_mask=True, use_custom_reward=True):
        """
        initial_values: List of 13 numerical values,
          1st element: Initial position (center distance),
          Remaining 12 elements: Other state/environment parameters.
        stop_threshold: Number of consecutive non-improving steps before stopping the episode.
        use_visited_mask: Whether to enable the visited mask mechanism.
        use_custom_reward: Whether to use customized reward shaping.
        """
        self.base_state = np.array(initial_values, dtype=np.float32)  # shape (13,)
        self.state = np.copy(self.base_state)
        self.current_position = self.base_state[0]
        
        # Build discrete action space
        self.action_space = np.arange(lower_bound, upper_bound + 1e-10, step_size)
        self.action_space = np.round(self.action_space, 3)
        self.action_dim = len(self.action_space)
        
        self.use_visited_mask = use_visited_mask
        self.use_custom_reward = use_custom_reward
        self.visited_mask = None

        # Precompute RF predictions for all candidate actions/positions
        self.precomputed = {}
        for pos in self.action_space:
            key = round(float(pos), 3)
            self.precomputed[key] = self.predict_at(pos)
        
        # Compute global minimum point information
        all_positions = [self.base_state[0]] + [pos for pos in self.action_space if not np.isclose(pos, self.base_state[0])]
        all_cmaxs = [self.predict_at(self.base_state[0])] + [
            self.precomputed[round(float(pos), 3)] for pos in self.action_space if not np.isclose(pos, self.base_state[0])
        ]
        self.overall_min_val = min(all_cmaxs)
        overall_min_index = all_cmaxs.index(self.overall_min_val)
        self.overall_min_position = all_positions[overall_min_index]
        
        # Track agent trajectory and convergence status
        self.current_best = None
        self.no_improve_steps = 0
        self.stop_threshold = stop_threshold

    def get_state(self):
        if self.use_visited_mask:
            if self.visited_mask is None:
                self.visited_mask = np.zeros(self.action_dim, dtype=np.float32)
            return np.concatenate((self.state, self.visited_mask))
        else:
            return self.state
    
    def reset(self):
        self.state = np.copy(self.base_state)
        self.current_position = self.base_state[0]
        if self.use_visited_mask:
            self.visited_mask = np.zeros(self.action_dim, dtype=np.float32)
            init_idx = np.where(np.isclose(self.action_space, self.state[0]))[0]
            if len(init_idx) > 0:
                self.visited_mask[init_idx[0]] = 1.0
        self.current_best = self.predict_at(self.state[0])
        self.no_improve_steps = 0
        return self.get_state()
    
    def step(self, action_index):
        new_position = self.action_space[action_index]
        if self.use_visited_mask:
            if self.visited_mask[action_index] == 1.0:
                return self.get_state(), -100.0, True, {}
        new_value = self.predict_at(new_position)
        
        improvement_coef = 20
        no_improve_penalty_coef = 25
        improvement = self.current_best - new_value

        if improvement > 0:
            improvement_reward = improvement * improvement_coef if self.use_custom_reward else 0
            no_improve_penalty = 0
            self.current_best = new_value
            self.no_improve_steps = 0
        else:
            improvement_reward = 0
            no_improve_penalty = abs(improvement) * no_improve_penalty_coef if self.use_custom_reward else 0
            self.no_improve_steps += 1

        step_cost = 15
        reward = -step_cost + improvement_reward - no_improve_penalty

        if self.no_improve_steps >= self.stop_threshold:
            reward -= 50
            if self.use_visited_mask:
                self.state[0] = new_position
                self.visited_mask[action_index] = 1.0
            return self.get_state(), reward, True, {}

        self.state[0] = new_position
        if self.use_visited_mask:
            self.visited_mask[action_index] = 1.0
        self.current_position = new_position
        done = False
        if self.use_visited_mask:
            done = np.all(self.visited_mask == 1.0)
        return self.get_state(), reward, done, {}

    def predict_at(self, pos):
        current_params = self.base_state[1:]
        input_data = pd.DataFrame({
            'CollectTargetXY':        [pos],
            'CWFiValue':              [current_params[0]],
            'CCWFiValue':             [current_params[1]],
            'CWfiiValue':             [current_params[2]],
            'CCWfiiValue':            [current_params[3]],
            'CWflValue':              [current_params[4]],
            'CCWflValue':             [current_params[5]],
            'CWfkValue':              [current_params[6]],
            'CCWfkValue':             [current_params[7]],
            'BacklashMeanValue':      [current_params[8]],
            'BacklashDiffValue':      [current_params[9]],
            'BacklashMaxValue':       [current_params[10]],
            'BacklashMinValue':       [current_params[11]],
            'input_CollectTargetXY':  [pos]
        })
        predictions = optimized_rf_model.predict(input_data)
        p = predictions[0]
        current_max = max(p[0], p[1], p[4], p[5])
        return current_max

# ------------------------------
# 4. Neural Network Architecture Definition
# ------------------------------
# Residual block with LayerNorm and ReLU
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.LayerNorm(features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.LayerNorm(features)
        )
        
    def forward(self, x):
        out = self.block(x)
        out += x
        return F.relu(out)

# (1) Basic DQN (Standard DQN / Double DQN) with Residual Connection and Regularization
class BasicDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BasicDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.bn1 = nn.LayerNorm(256)
        
        # Residual block
        self.res_block = ResidualBlock(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.LayerNorm(128)
        
        # Dropout regularization
        self.dropout = nn.Dropout(0.2)
        
        # Output layer
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.res_block(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

# (2) Dueling DQN with Residual Connection and Regularization
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        # Feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            ResidualBlock(256)
        )
        # Value stream V(s)
        self.value_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        # Advantage stream A(s, a)
        self.advantage_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_layer(features)
        advantage = self.advantage_layer(features)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

# ------------------------------
# 5. Replay Buffer and Expert Demonstration
# ------------------------------
# Prioritized Experience Replay (PER)
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        epsilon = 1e-6
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio) + epsilon

    def __len__(self):
        return len(self.buffer)

# Uniform Replay Buffer (without priority)
class SimpleReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        weights = np.ones(len(indices))
        return states, actions, rewards, next_states, dones, indices, weights
    def update_priorities(self, batch_indices, batch_priorities):
        pass
    def __len__(self):
        return len(self.buffer)

def add_expert_demo(env, replay_buffer, use_expert_demo=True):
    if not use_expert_demo:
        return
    demo_state = env.reset()
    candidate_actions = []
    for i, pos in enumerate(env.action_space):
        if env.use_visited_mask:
            if env.visited_mask[i] == 0:
                pred_val = env.predict_at(pos)
                candidate_actions.append((pred_val, i))
        else:
            pred_val = env.predict_at(pos)
            candidate_actions.append((pred_val, i))
    if candidate_actions:
        candidate_actions.sort(key=lambda x: x[0])
        expert_action = candidate_actions[0][1]
    else:
        expert_action = 0
    demo_next_state, demo_reward, demo_done, _ = env.step(expert_action)
    replay_buffer.push(demo_state, expert_action, demo_reward, demo_next_state, demo_done)
    print("Expert demonstration transition added to replay buffer.")

# ------------------------------
# 6. Universal Training Function (Supports DQN, Double DQN, and Dueling DQN)
# ------------------------------
def train_dqn(env, num_episodes=30, batch_size=128, gamma=0.90, 
              learning_rate=1e-3, target_update=5, replay_capacity=1000000,
              epsilon_final=0.01, alpha_epsilon=0.1, fast_decay_rate=0.05, slow_decay_rate=0.01,
              model_variant="DQN", initial_model=None, initial_replay_buffer=None,
              use_dynamic_epsilon=True, use_expert_demo=True, use_prioritized_replay=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.get_state().shape[0]
    action_dim = len(env.action_space)
    
    # Model selection
    if model_variant == "DuelingDQN":
        if initial_model is not None:
            policy_net = initial_model.to(device)
        else:
            policy_net = DuelingDQN(state_dim, action_dim).to(device)
    else:  # Applicable to "DQN" and "DoubleDQN"
        if initial_model is not None:
            policy_net = initial_model.to(device)
        else:
            policy_net = BasicDQN(state_dim, action_dim).to(device)
    
    # Replay Buffer selection
    if initial_replay_buffer is not None:
        replay_buffer = initial_replay_buffer
    else:
        if use_prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(replay_capacity, alpha=0.6)
        else:
            replay_buffer = SimpleReplayBuffer(replay_capacity)
        
    # Add expert demonstration if enabled
    add_expert_demo(env, replay_buffer, use_expert_demo=use_expert_demo)
    
    # Target network initialization
    if model_variant == "DuelingDQN":
        target_net = DuelingDQN(state_dim, action_dim).to(device)
    else:
        target_net = BasicDQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    # Initial epsilon
    epsilon = 0.8
    moving_avg_reward = 0.0
    epsilon_reset_counter = 0
    all_rewards = []
    losses = []
    last_episode_trajectory = []
    consecutive_success = 0
    consecutive_success_threshold = 5
    final_episode = num_episodes - 1

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        episode_trajectory = [state[0]]
        while not done:
            if random.random() < epsilon:
                action_idx = random.randrange(action_dim)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action_idx = q_values.max(1)[1].item()
            next_state, reward, done, _ = env.step(action_idx)
            episode_reward += reward
            replay_buffer.push(state, action_idx, reward, next_state, done)
            state = next_state
            episode_trajectory.append(state[0])

            if len(replay_buffer) > batch_size:
                if use_prioritized_replay:
                    beta = 0.4
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done, indices, weights = replay_buffer.sample(batch_size, beta)
                else:
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done, indices, weights = replay_buffer.sample(batch_size)
                batch_state = torch.FloatTensor(batch_state).to(device)
                batch_action = torch.LongTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                batch_next_state = torch.FloatTensor(batch_next_state).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)
                weights = torch.FloatTensor(weights).unsqueeze(1).to(device)

                current_q_values = policy_net(batch_state).gather(1, batch_action)
                
                # Double DQN action selection vs evaluation
                if model_variant == "DoubleDQN":
                    next_actions = policy_net(batch_next_state).max(1)[1].unsqueeze(1)
                    next_q_values = target_net(batch_next_state).gather(1, next_actions)
                else:
                    next_q_values = target_net(batch_next_state).max(1)[0].unsqueeze(1)
                expected_q_values = batch_reward + gamma * next_q_values * (1 - batch_done)

                td_errors = current_q_values - expected_q_values.detach()
                loss = (F.mse_loss(current_q_values, expected_q_values.detach(), reduction='none') * weights).mean()
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if use_prioritized_replay:
                    new_priorities = td_errors.abs().detach().cpu().numpy().squeeze()
                    replay_buffer.update_priorities(indices, new_priorities)
        steps_in_episode = len(episode_trajectory) - 1

        if episode_reward == 100 and steps_in_episode == 1 and epsilon == epsilon_final:
            consecutive_success += 1
        else:
            consecutive_success = 0

        print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Steps: {steps_in_episode}, "
              f"Consec. Success: {consecutive_success}, moving_avg_reward: {moving_avg_reward:.2f}, epsilon: {epsilon:.3f}")

        # Insert expert demonstration every 10 episodes if enabled
        if episode % 10 == 0 and episode != 0 and use_expert_demo:
            add_expert_demo(env, replay_buffer, use_expert_demo=use_expert_demo)
            print("Expert demonstration transition inserted at 10-episode interval.")

        # Dynamic epsilon scheduling
        if use_dynamic_epsilon:
            moving_avg_reward = (1 - alpha_epsilon) * moving_avg_reward + alpha_epsilon * episode_reward
            if moving_avg_reward > 0:
                epsilon = max(epsilon_final, epsilon - fast_decay_rate)
            else:
                epsilon = max(epsilon_final, epsilon - slow_decay_rate)
            if epsilon == epsilon_final:
                epsilon_reset_counter += 1
            else:
                epsilon_reset_counter = 0
            if epsilon_reset_counter >= 50:
                epsilon = 0.6
                epsilon_reset_counter = 0
                print("Epsilon reset to 0.6")

        all_rewards.append(episode_reward)
        last_episode_trajectory = episode_trajectory
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if consecutive_success >= consecutive_success_threshold:
            print("Early stopping triggered: Consecutive success count reached threshold.")
            final_episode = episode
            break

    early_stop_flag = (final_episode < num_episodes - 1)
    return policy_net, all_rewards, losses, last_episode_trajectory, replay_buffer, early_stop_flag

# ------------------------------
# 7. Main Program Entry
# ------------------------------
if __name__ == "__main__":
    # File paths for saving Replay Buffer and model weights
    replay_buffer_save_path = r"Complete_model\duelingDQN\replay_buffer_ablation.pkl"
    model_save_path = r"Complete_model\duelingDQN\ablation_model.pt"

    # Select execution mode and model variant
    mode = input("Select execution mode (train: t, predict: p): ").strip().lower()
    model_variant_input = input("Select model architecture (dqn / doubledqn / duelingdqn): ").strip().lower()
    if model_variant_input == "dqn":
        model_variant = "DQN"
    elif model_variant_input == "doubledqn":
        model_variant = "DoubleDQN"
    elif model_variant_input == "duelingdqn":
        model_variant = "DuelingDQN"
    else:
        print("Invalid model variant input!")
        exit()

    # Module selection for ablation study
    # 0: Disable all enhancements
    # 1: Environment Design (Visited Mask)
    # 2: Customized Reward Shaping
    # 3: Expert Demonstration
    # 4: Prioritized Experience Replay (PER)
    # 5: Dynamic Epsilon Adjustment
    func_input = input("Select features to enable, e.g., 1,2,3 (0: Disable all / 1: Visited Mask / 2: Reward Shaping / 3: Expert Demo / 4: PER / 5: Dynamic Epsilon): ").strip()
    
    use_visited_mask = False
    use_custom_reward = False
    use_expert_demo = False
    use_prioritized_replay = False
    use_dynamic_epsilon = False
    if func_input == "0":
        pass
    else:
        funcs = [s.strip() for s in func_input.split(",")]
        if "1" in funcs:
            use_visited_mask = True
        if "2" in funcs:
            use_custom_reward = True
        if "3" in funcs:
            use_expert_demo = True
        if "4" in funcs:
            use_prioritized_replay = True
        if "5" in funcs:
            use_dynamic_epsilon = True

    print("Configuration:")
    print(f"Environment Design (Visited Mask): {use_visited_mask}")
    print(f"Customized Reward Shaping: {use_custom_reward}")
    print(f"Expert Demonstration: {use_expert_demo}")
    print(f"Prioritized Experience Replay (PER): {use_prioritized_replay}")
    print(f"Dynamic Epsilon Adjustment: {use_dynamic_epsilon}")

    if mode == 't':
        # Training Mode
        training_folder = r"data_set_RF_DD_train"
        csv_files = glob.glob(os.path.join(training_folder, "*.csv"))
        if len(csv_files) == 0:
            print("No CSV files found. Please verify the folder path.")
            exit()
            
        use_existing = input("Use existing replay buffer and model weights? (y/n): ").strip().lower()
        def is_replay_buffer_format_consistent(replay_buffer_obj, env):
            if len(replay_buffer_obj.buffer) == 0:
                return True
            sample_transition = replay_buffer_obj.buffer[0]
            state, action, reward, next_state, done = sample_transition
            expected_state_shape = env.get_state().shape
            if state.shape != expected_state_shape:
                return False
            return True

        if use_existing == 'y':
            if os.path.exists(replay_buffer_save_path):
                with open(replay_buffer_save_path, "rb") as f:
                    loaded_replay_buffer_obj = pickle.load(f)
                if isinstance(loaded_replay_buffer_obj, list):
                    print("Detected replay buffer saved as list; converting to PrioritizedReplayBuffer.")
                    capacity = max(1000000, len(loaded_replay_buffer_obj))
                    new_buffer = PrioritizedReplayBuffer(capacity, alpha=0.6)
                    new_buffer.buffer = loaded_replay_buffer_obj
                    new_buffer.pos = len(loaded_replay_buffer_obj) % capacity
                    new_buffer.priorities[:len(loaded_replay_buffer_obj)] = 1.0
                    loaded_replay_buffer_obj = new_buffer
                dummy_env = SimulationEnv([0.0]*13, use_visited_mask=use_visited_mask, use_custom_reward=use_custom_reward)
                if is_replay_buffer_format_consistent(loaded_replay_buffer_obj, dummy_env):
                    print("Replay buffer format matches; loaded existing replay buffer.")
                    loaded_replay_buffer = loaded_replay_buffer_obj
                else:
                    print("Replay buffer format mismatch; initializing a new replay buffer.")
                    loaded_replay_buffer = None
            else:
                print("Replay buffer file not found; initializing a new replay buffer.")
                loaded_replay_buffer = None

            if os.path.exists(model_save_path):
                dummy_env = SimulationEnv([0.0]*13, use_visited_mask=use_visited_mask, use_custom_reward=use_custom_reward)
                state_dim = dummy_env.get_state().shape[0]
                action_dim = dummy_env.action_dim
                if model_variant == "DuelingDQN":
                    loaded_model = DuelingDQN(state_dim, action_dim)
                else:
                    loaded_model = BasicDQN(state_dim, action_dim)
                loaded_model.load_state_dict(torch.load(model_save_path, map_location=torch.device("cpu")))
                loaded_model.eval()
                print("Model weights loaded successfully!")
            else:
                print("Model file not found; initializing a new model.")
                loaded_model = None
        else:
            loaded_replay_buffer = None
            loaded_model = None

        trained_policy = loaded_model
        replay_buffer_obj = loaded_replay_buffer
        all_rewards_all = []
        last_trajectory = None
        not_early_stop_list = []

        for csv_index, csv_file in enumerate(csv_files, start=1):
            print(f"[{csv_index}/{len(csv_files)}] Starting training on file: {csv_file}")
            df = pd.read_csv(csv_file)
            total = len(df)
            print(f"Total samples in file: {total}")
            columns = ["CollectTargetXY", "CWFiValue", "CCWFiValue", "CWfiiValue", "CCWfiiValue", 
                       "CWflValue", "CCWflValue", "CWfkValue", "CCWfkValue", 
                       "BacklashMeanValue", "BacklashDiffValue", "BacklashMaxValue", "BacklashMinValue"]
            for i, (idx, row) in enumerate(df.iterrows(), start=1):
                print(f"[{csv_index}/{len(csv_files)}],[{i}/{total}]")
                if not all(col in row.index for col in columns):
                    print("CSV columns do not match expected format; skipping row.")
                    continue
                initial_values = [float(row[col]) for col in columns]
                print(f"Initial state parameters: {initial_values}")
                env = SimulationEnv(initial_values, use_visited_mask=use_visited_mask, use_custom_reward=use_custom_reward)
                trained_policy, rewards, loss_list, last_episode_trajectory, replay_buffer_obj, early_stop_flag = train_dqn(
                    env,
                    num_episodes=50,
                    initial_model=trained_policy,
                    initial_replay_buffer=replay_buffer_obj,
                    model_variant=model_variant,
                    use_dynamic_epsilon=use_dynamic_epsilon,
                    use_expert_demo=use_expert_demo,
                    use_prioritized_replay=use_prioritized_replay
                )
                all_rewards_all.extend(rewards)
                last_trajectory = last_episode_trajectory

                if not early_stop_flag:
                    not_early_stop_list.append((csv_file, i, initial_values))
        
        torch.save(trained_policy.state_dict(), model_save_path)
        with open(replay_buffer_save_path, "wb") as f:
            pickle.dump(replay_buffer_obj, f)
        print("Training complete. Model weights and replay buffer saved.")
        
        total_data = 0
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            total_data += len(df)
        print(f"Non-early-stopped episodes / Total episodes: {len(not_early_stop_list)}/{total_data}")
        if len(not_early_stop_list) > 0:
            print("Initial parameters for non-early-stopped episodes:")
            for entry in not_early_stop_list:
                csv_file, row_index, init_vals = entry
                print(f"{os.path.splitext(os.path.basename(csv_file))[0]}, {init_vals}")

        trajectory_positions = last_trajectory
        trajectory_values = [env.precomputed[round(float(pos), 3)] for pos in trajectory_positions]
        plt.figure(figsize=(10, 6))
        plt.plot(trajectory_positions, trajectory_values, color='green', linestyle='-', marker='o', label='Last Episode Trajectory')
        plt.xlabel('Position')
        plt.ylabel('Current Max')
        plt.title('Agent Action Trajectory in the Final Episode')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif mode == 'p':
        # Prediction / Inference Mode
        input_folder = r"data_set_RF_DD_test"
        input_files = glob.glob(os.path.join(input_folder, "*.csv"))
        if not input_files:
            print(f"No CSV files found in directory: {input_folder}")
            exit()
        
        output_data = []
        success_count = 0
        success_steps = []
        fail_count = 0
        fail_squared_errors = []
        success_squared_errors = []
        failure_steps = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_save_path):
            print(f"Model file not found: {model_save_path}. Cannot perform inference.")
            exit()
        dummy_env = SimulationEnv([0.0]*13, use_visited_mask=use_visited_mask, use_custom_reward=use_custom_reward)
        state_dim = dummy_env.get_state().shape[0]
        action_dim = dummy_env.action_dim
        if model_variant == "DuelingDQN":
            model = DuelingDQN(state_dim, action_dim)
        else:
            model = BasicDQN(state_dim, action_dim)
        model.load_state_dict(torch.load(model_save_path, map_location=torch.device("cpu")))
        model = model.to(device)
        model.eval()
        print("Model weights loaded successfully!")
        
        def predict_minimum_point(env, model, device):
            state = env.reset()
            done = False
            trajectory = [state[0]]
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action_index = q_values.max(1)[1].item()
                next_state, reward, done, _ = env.step(action_index)
                trajectory.append(next_state[0])
                state = next_state
            best_point = trajectory[0]
            best_val = env.precomputed[round(float(best_point), 3)]
            for pos in trajectory:
                pos_val = env.precomputed[round(float(pos), 3)]
                if pos_val < best_val:
                    best_val = pos_val
                    best_point = pos
            return best_point, trajectory

        for input_file in input_files:
            print(f"Processing file: {input_file}")
            try:
                df = pd.read_csv(input_file)
            except Exception as e:
                print(f"Error reading file {input_file}: {e}")
                continue
            columns = ["CollectTargetXY", "CWFiValue", "CCWFiValue", "CWfiiValue", "CCWfiiValue", 
                       "CWflValue", "CCWflValue", "CWfkValue", "CCWfkValue", 
                       "BacklashMeanValue", "BacklashDiffValue", "BacklashMaxValue", "BacklashMinValue"]
            for idx, row in df.iterrows():
                try:
                    initial_values = [float(row[col]) for col in columns]
                except Exception as e:
                    print(f"Error reading row {idx} in file {input_file}: {e}")
                    continue
                env = SimulationEnv(initial_values, use_visited_mask=use_visited_mask, use_custom_reward=use_custom_reward)
                predicted_min, trajectory = predict_minimum_point(env, model, device)
                steps = len(trajectory) - 1
                overall_min = env.overall_min_position
                overall_min_val = env.overall_min_val
                
                output_data.append({
                    "input_file": os.path.basename(input_file),
                    "row_index": idx,
                    "initial_parameters": initial_values,
                    "overall_min_position": overall_min,
                    "overall_min_val": overall_min_val,
                    "predicted_min_position": predicted_min,
                    "steps": steps,
                    "trajectory": trajectory
                })
                
                if round(abs(predicted_min - overall_min), 2) <= 0.01:
                    success_count += 1
                    success_steps.append(steps)
                    success_squared_errors.append((predicted_min - overall_min) ** 2)
                else:
                    fail_count += 1
                    fail_squared_errors.append((predicted_min - overall_min) ** 2)
                    failure_steps.append(steps)
                
                initial_point = env.base_state[0]
                initial_current_max = env.predict_at(initial_point)
                global_positions = []
                global_cmaxs = []
                for pos in env.action_space:
                    if np.isclose(pos, initial_point):
                        continue
                    key = round(float(pos), 3)
                    cmax = env.precomputed[key]
                    global_positions.append(pos)
                    global_cmaxs.append(cmax)
                plt.figure(figsize=(10, 6))
                plt.scatter(global_positions, global_cmaxs, color='orange', label='Global Sampling Points')
                plt.scatter([initial_point], [initial_current_max], color='blue', label='User Input')
                plt.scatter([overall_min], [overall_min_val], marker='*', color='purple', s=150, label='Overall Min Point')
                plt.annotate(f"Min: ({overall_min:.3f}, {overall_min_val:.3f})",
                            xy=(overall_min, overall_min_val),
                            xytext=(overall_min, overall_min_val - 0.05*(max(global_cmaxs+[initial_current_max]) - min(global_cmaxs+[initial_current_max]))),
                            arrowprops=dict(facecolor='purple', arrowstyle='->'),
                            fontsize=9,
                            color='purple')
                trajectory_values = [env.precomputed[round(float(pos), 3)] for pos in trajectory]
                plt.plot(trajectory, trajectory_values, color='green', linestyle='-', marker='o', label='Trajectory')
                for tidx, (tpos, tval) in enumerate(zip(trajectory, trajectory_values)):
                    plt.text(tpos, tval, f'{tidx}', fontsize=9, color='black', ha='center', va='bottom')
                plt.xlabel('Position')
                plt.ylabel('Current Max')
                plt.title('Global Sampling & Trajectory from DQN Prediction')
                plt.legend()
                plt.grid(True)
                fig_filename = os.path.join(input_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_row{idx}_prediction.png")
                plt.savefig(fig_filename)
                plt.close()
        
        output_df = pd.DataFrame(output_data)
        output_csv_path = os.path.join(input_folder, "prediction_results.csv")
        output_df.to_csv(output_csv_path, index=False)
        
        summary = {}
        summary["success_count"] = success_count
        summary["failure_count"] = fail_count
        summary["average_success_steps"] = np.mean(success_steps) if success_count > 0 else None
        summary["rmse_success"] = np.sqrt(np.mean(success_squared_errors)) if success_count > 0 else None
        summary["rmse_failure"] = np.sqrt(np.mean(fail_squared_errors)) if fail_count > 0 else None
        success_steps_distribution = dict(Counter(success_steps)) if success_steps else {}
        failure_steps_distribution = dict(Counter(failure_steps)) if failure_steps else {}
        average_failure_steps = np.mean(failure_steps) if failure_steps else None
        summary["success_steps_distribution"] = success_steps_distribution
        summary["failure_steps_distribution"] = failure_steps_distribution
        summary["average_failure_steps"] = average_failure_steps

        columns_order = [
            "success_count",
            "success_steps_distribution",
            "average_success_steps",
            "rmse_success",
            "failure_count",
            "failure_steps_distribution",
            "average_failure_steps",
            "rmse_failure"
        ]
        summary_df = pd.DataFrame([summary], columns=columns_order)
        summary_csv_path = os.path.join(input_folder, "prediction_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print("Inference completed. Results saved to:", input_folder)
    else:
        print("Invalid input. Please enter 't' (train) or 'p' (predict)!")