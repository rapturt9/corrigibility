import numpy as np
from minigrid_power_env import PowerGridEnv # Import custom environment
from iql_agent import IQLPowerAgent # Import adapted agent
import matplotlib.pyplot as plt

def main():
    # --- Environment Setup ---
    env_config = {
        "size": 7,
        "agent_start_pos": (1, 1),
        # Let human start/goal be random for more variation, or fix them
        "human_start_pos": None, # (5, 5),
        "human_goal_pos": None, # (1, 5),
        "max_steps": 100,
        "render_mode": "rgb_array" # Use "human" for interactive view
    }
    env = PowerGridEnv(**env_config)
    # Extract action space size (MiniGrid uses Discrete space)
    # 0: right, 1: down, 2: left, 3: up, 4: pickup, 5: drop, 6: toggle
    # We only need movement actions for this simple agent/env
    action_map = {
        0: 3, # Agent Up -> Env Action 3 (Up)
        1: 1, # Agent Down -> Env Action 1 (Down)
        2: 2, # Agent Left -> Env Action 2 (Left)
        3: 0, # Agent Right -> Env Action 0 (Right)
        # 4: 6, # Agent Wait -> Env Action 6 (Toggle/Wait - adapt if needed)
        # For simplicity, let's map agent actions 0-3 to env movements 3,1,2,0
    }
    agent_action_space_size = len(action_map) # Only 4 movement actions

    # --- Agent Setup ---
    # Define the goal set G the ROBOT considers for power calculation
    # These should be possible goal positions in the grid
    # Example: corners (excluding agent start if fixed)
    grid_size = env_config["size"] # Use size from config
    potential_goals = [
        (1, grid_size - 2), # Top-right
        (grid_size - 2, 1), # Bottom-left
        (grid_size - 2, grid_size - 2) # Bottom-right
    ]
    goal_set = [g for g in potential_goals if g != env_config["agent_start_pos"]]
    if not goal_set: # Handle case where all potential goals overlap start
        goal_set = [(grid_size // 2, grid_size // 2)] # Default to center
    num_goals = len(goal_set)
    goal_prior = {i: 1.0/num_goals for i in range(num_goals)} # Uniform prior

    # Power function f(z)
    f_func = lambda z: 2.0 - 2.0 / (z + 1e-6) # Recommended in paper

    agent = IQLPowerAgent(
        action_space_size=agent_action_space_size, # Agent uses 4 actions
        goal_set=goal_set,
        goal_prior=goal_prior,
        gamma_h=0.95,
        gamma_r=0.99,
        alpha_h=0.1,
        alpha_r=0.05,
        beta_h=2.0,
        epsilon_r=1.0,
        epsilon_r_decay=0.9995,
        epsilon_r_min=0.05,
        f_func=f_func,
        eta=0.0
    )

    # --- Training Loop ---
    num_episodes = 20000
    print_every = 1000
    render_every = 0 # Set > 0 to render episodes, e.g., 5000

    total_rewards_log = []
    human_success_log = []

    print(f"Starting training for {num_episodes} episodes...")
    print(f"Environment: {env.spec.id if env.spec else 'Custom PowerGridEnv'}")
    print(f"Grid Size: {env.width}x{env.height}") # Use env.width and env.height
    print(f"Robot considers Goal Set: {agent.goal_set}")
    print(f"Robot Goal Prior (indices): {agent.goal_prior}")

    for episode in range(num_episodes):
        obs, info = env.reset()
        # Get actual human goal for this episode from info
        actual_human_goal_pos = tuple(info['goal_pos'])
        try:
            # Agent needs the *index* of the human's current goal relative to its goal_set
            human_current_goal_idx = agent.goal_indices[actual_human_goal_pos]
        except KeyError:
            # This shouldn't happen if goal_set includes all possible env goals
            print(f"Warning: Actual goal {actual_human_goal_pos} not in agent's goal set {agent.goal_set}. Using index 0.")
            human_current_goal_idx = 0 # Fallback

        # Extract current positions for state tuple
        agent_pos = tuple(env.agent_pos) # agent_pos is tracked by MiniGridEnv
        human_pos = tuple(info['human_pos'])

        terminated = False
        truncated = False
        episode_reward = 0
        human_succeeded = False

        while not terminated and not truncated:
            # 1. Robot chooses action (0-3)
            robot_agent_action = agent.choose_robot_action(agent_pos, human_pos)
            # Map agent action to environment action
            robot_env_action = action_map.get(robot_agent_action)
            if robot_env_action is None:
                 print(f"Warning: Invalid agent action {robot_agent_action}, using wait/toggle.")
                 robot_env_action = 6 # Default to toggle/wait

            # 2. Simulate environment step
            next_obs, robot_reward, reward_h_obs, terminated, truncated, info = env.step(robot_env_action)

            # 3. Extract next state info
            next_agent_pos = tuple(env.agent_pos)
            next_human_pos = tuple(info['human_pos'])

            # 4. Agent Update
            # Simulate human action based on agent's internal model for Q_h update
            simulated_human_action = agent.get_human_action_for_simulation(agent_pos, human_pos, human_current_goal_idx)
            # Map simulated human action (0-3) to env action if needed, though not strictly necessary for update
            # simulated_human_env_action = action_map.get(simulated_human_action)

            agent.update(agent_pos, human_pos,
                         robot_agent_action, simulated_human_action, # Use agent's action indices
                         reward_h_obs,
                         next_agent_pos, next_human_pos,
                         human_current_goal_idx,
                         terminated or truncated) # Done if terminated or truncated

            # 5. Prepare for next step
            agent_pos = next_agent_pos
            human_pos = next_human_pos
            episode_reward += robot_reward
            if reward_h_obs > 0: # If human reached goal this step
                 human_succeeded = True

            # Optional Rendering
            if render_every > 0 and episode % render_every == 0:
                 frame = env.render()
                 # Display frame if needed (e.g., using matplotlib or cv2)
                 # plt.imshow(frame)
                 # plt.pause(0.1) # Adjust pause duration
                 # plt.clf()


        total_rewards_log.append(episode_reward)
        human_success_log.append(1 if human_succeeded else 0)

        # Print progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(total_rewards_log[-print_every:])
            success_rate = np.mean(human_success_log[-print_every:]) * 100
            print(f"Episode: {episode + 1}/{num_episodes} | Avg Reward (last {print_every}): {avg_reward:.3f} | Success Rate: {success_rate:.1f}% | Epsilon_r: {agent.epsilon_r:.3f}")

    print("Training finished.")
    env.close()

    # --- Optional: Plotting ---
    plt.figure(figsize=(12, 5))
    window_size = 100 # Rolling window size

    plt.subplot(1, 2, 1)
    if len(total_rewards_log) >= window_size:
        moving_avg = np.convolve(total_rewards_log, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size -1, len(total_rewards_log)), moving_avg)
    else:
        plt.plot(total_rewards_log) # Plot raw data if not enough points for window
    plt.title(f'Robot Cumulative Reward (Moving Avg, Window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')

    plt.subplot(1, 2, 2)
    if len(human_success_log) >= window_size:
        moving_avg_success = np.convolve(human_success_log, np.ones(window_size)/window_size, mode='valid') * 100
        plt.plot(np.arange(window_size-1, len(human_success_log)), moving_avg_success)
    else:
         plt.plot(np.array(human_success_log)*100)
    plt.title(f'Human Success Rate (Moving Avg, Window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 105)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
