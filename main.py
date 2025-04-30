# Tool code to modify main.py
import numpy as np
from minigrid_power_env import PowerGridEnv # Import custom environment
from iql_agent import IQLPowerAgent # Import adapted agent
import matplotlib.pyplot as plt
import time # Import time for pausing

def main():
    env_config = {
        "size": 7,
        "agent_start_pos": (1, 1),
        # Human start/goal are handled internally by env now
        # "human_start_pos": None,
        # "human_goal_pos": None,
        "max_steps": 100,
        "render_mode": "human" # Use "human" for interactive view
    }
    env = PowerGridEnv(**env_config)

    # Action mapping (Agent action -> Env action)
    # Agent actions: 0: Up, 1: Down, 2: Left, 3: Right (semantic)
    # Env actions: 0: right, 1: down, 2: left, 3: up, ...
    action_map = {
        0: 3, # Agent Up    -> Env Action 3 (Up)
        1: 1, # Agent Down  -> Env Action 1 (Down)
        2: 2, # Agent Left  -> Env Action 2 (Left)
        3: 0, # Agent Right -> Env Action 0 (Right)
    }
    agent_action_space_size = len(action_map) # Agent uses 4 actions

    # --- Agent Setup ---\n    # Get the goal set the environment uses
    goal_set = env.possible_goal_positions # Use the goals defined in the env
    num_goals = len(goal_set)
    if num_goals == 0:
        raise ValueError("Environment's possible_goal_positions is empty!")
    goal_prior = {i: 1.0/num_goals for i in range(num_goals)} # Uniform prior

    # Power function f(z)
    f_func = lambda z: 2.0 - 2.0 / (z + 1e-6) # Recommended in paper

    agent = IQLPowerAgent(
        action_space_size=agent_action_space_size, # Agent uses 4 actions
        goal_set=goal_set, # Use the actual goals from the env
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

    # --- Training Loop ---\n    
    num_episodes = 20000
    print_every = 1000
    render_every = 100 # Set > 0 to render episodes, e.g., 100
    render_pause = 0.05 # Pause duration in seconds when rendering (reduced)

    total_rewards_log = []
    human_success_log = []

    print(f"Starting training for {num_episodes} episodes...")
    print(f"Environment: Custom PowerGridEnv")
    print(f"Grid Size: {env.width}x{env.height}")
    print(f"Agent considers Goal Set G: {agent.goal_set}")
    # print(f"Agent Goal Prior (indices): {agent.goal_prior}") # Less informative

    for episode in range(num_episodes):
        obs, info = env.reset()

        # Extract state components from the dictionary observation
        agent_pos = obs['agent_pos']
        human_pos = obs['human_pos']
        actual_human_goal_pos = info['goal_pos'] # Goal pos comes from info

        # Find the index of the actual goal within the agent's known goal set
        try:
            human_current_goal_idx = agent.goal_indices[actual_human_goal_pos]
        except KeyError:
            # This should NOT happen now if agent uses env.possible_goal_positions
            print(f"FATAL ERROR: Actual goal {actual_human_goal_pos} not in agent's goal set {agent.goal_set}. Check env/agent goal consistency.")
            # Fallback or raise error? Let's raise for debugging.
            raise ValueError(f"Actual goal {actual_human_goal_pos} not in agent's goal set {agent.goal_set}")
            # human_current_goal_idx = 0 # Fallback

        terminated = False
        truncated = False
        episode_reward = 0
        human_succeeded_in_episode = False
        step_count = 0 # Track steps within episode if needed

        while not terminated and not truncated:
            step_count += 1
            # 1. Robot chooses action (semantic: 0-Up, 1-Down, 2-Left, 3-Right)
            robot_agent_action = agent.choose_robot_action(agent_pos, human_pos)

            # Map agent action to environment action
            robot_env_action = action_map.get(robot_agent_action)
            if robot_env_action is None:
                 print(f"Warning: Invalid agent action {robot_agent_action}, defaulting to env action 0 (right).")
                 robot_env_action = 0 # Default to right move

            # 2. Simulate environment step
            # Env returns: obs_dict, robot_reward, reward_h_obs, terminated, truncated, info_dict
            next_obs, robot_reward, reward_h_obs, terminated, truncated, info = env.step(robot_env_action)

            # 3. Extract next state info from the NEW observation dictionary
            next_agent_pos = next_obs['agent_pos']
            next_human_pos = next_obs['human_pos']
            # Goal position doesn't change within an episode
            # next_goal_pos = next_obs['goal_pos'] # or info['goal_pos']

            # 4. Agent Update
            # Simulate human action based on agent's internal model for Q_h update
            simulated_human_action = agent.get_human_action_for_simulation(agent_pos, human_pos, human_current_goal_idx)

            agent.update(agent_pos, human_pos,
                         robot_agent_action, simulated_human_action, # Use agent's semantic action indices
                         reward_h_obs, # Use the base human reward from env
                         next_agent_pos, next_human_pos,
                         human_current_goal_idx,
                         terminated or truncated) # Done flag

            # 5. Prepare for next step
            agent_pos = next_agent_pos
            human_pos = next_human_pos
            episode_reward += robot_reward
            if reward_h_obs > 0: # If human reached goal this step (r_h_obs == 1.0)
                 human_succeeded_in_episode = True

            # Optional Rendering (render is called within env.step if mode is human)
            if render_every > 0 and episode % render_every == 0:
                 # env.render() # No longer needed here
                 time.sleep(render_pause) # Add a pause to see the frame


        total_rewards_log.append(episode_reward)
        human_success_log.append(1 if human_succeeded_in_episode else 0)

        # Print progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(total_rewards_log[-print_every:])
            success_rate = np.mean(human_success_log[-print_every:]) * 100
            print(f"Episode: {episode + 1}/{num_episodes} | Avg Reward (last {print_every}): {avg_reward:.3f} | Success Rate: {success_rate:.1f}% | Epsilon_r: {agent.epsilon_r:.3f}")

    print("Training finished.")
    env.close()

    # --- Optional: Plotting ---
    # (Plotting code remains the same)
    plt.figure(figsize=(12, 5))
    # ... (rest of plotting code) ...
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
