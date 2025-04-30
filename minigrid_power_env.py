import numpy as np
import gymnasium as gym # Import gymnasium
from gymnasium import spaces # Import spaces
import random # Import random for goal selection

from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX, IDX_TO_OBJECT
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, WorldObj, Ball # Using Ball to represent Human
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_rect # Ensure rendering utils are imported
from minigrid.core.constants import COLORS # Ensure COLORS are imported

# Define a new object type for the human NPC
# Find the next available index
next_idx = max(OBJECT_TO_IDX.values()) + 1
IDX_TO_OBJECT[next_idx] = "human"
OBJECT_TO_IDX["human"] = next_idx

class HumanNPC(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("human", color)

    def can_overlap(self):
        return True # Agent can potentially occupy the same cell

    def render(self, img):
        # Render as a blue ball
        fill_coords(img, point_in_rect(0.25, 0.75, 0.25, 0.75), COLORS[self.color])


class PowerGridEnv(MiniGridEnv):
    """
    MiniGrid environment with a Robot (agent) and a Human (NPC).
    Uses Dict observation space: {agent_pos, human_pos, goal_pos}.
    The agent's goal is implicitly to help the human reach its goal.
    """
    def __init__(
        self,
        size=7, # Default size consistent with main.py
        agent_start_pos=(1, 1),
        human_start_pos=None, # If None, random placement
        # human_goal_pos is now selected from possible_goal_positions
        max_steps=100, # Default max_steps consistent with main.py
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.human_start_pos = human_start_pos
        # self.human_goal_pos = human_goal_pos # Removed, goal is selected internally

        # Define the set of possible goals the environment can use
        # Consistent with the corners used in main.py initially
        self.possible_goal_positions = [
            (1, size - 2), # Top-right
            (size - 2, 1), # Bottom-left
            (size - 2, size - 2) # Bottom-right
        ]
        # Filter out agent start position if it conflicts
        self.possible_goal_positions = [
            p for p in self.possible_goal_positions if p != self.agent_start_pos
        ]
        if not self.possible_goal_positions: # Fallback if all goals conflict
             self.possible_goal_positions = [(size // 2, size // 2)]


        # Store human position and the goal for the current episode
        self._human_pos = None
        self._human_goal_pos = None # Set in _gen_grid

        mission_space = MissionSpace(mission_func=lambda: "Help the human reach the goal")

        # Define observation space (Dict)
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "human_pos": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "goal_pos": spaces.Box(0, size - 1, shape=(2,), dtype=int)
        })

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            # see_through_walls=True, # Not needed for Dict obs
            # agent_view_size=size,   # Not needed for Dict obs
            **kwargs,
        )
        # print(f"Initialized PowerGridEnv. Observation Space: {self.observation_space}")


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the grid walls
        self.grid.wall_rect(0, 0, width, height)

        # --- Placement Order: Goal -> Human -> Agent ---

        # 1. Place the human goal randomly from the possible set
        goal_idx = self.np_random.integers(0, len(self.possible_goal_positions))
        self._human_goal_pos = self.possible_goal_positions[goal_idx]
        self.put_obj(Goal(), self._human_goal_pos[0], self._human_goal_pos[1])
        # print(f"Placed goal at: {self._human_goal_pos}")

        # 2. Place the human NPC
        if self.human_start_pos is not None:
            # Check if specified start pos conflicts with goal
            if self.human_start_pos == self._human_goal_pos:
                 print(f"Warning: Human start pos {self.human_start_pos} conflicts with goal {self._human_goal_pos}. Placing randomly.")
                 start_pos = self.place_obj(HumanNPC(), max_tries=100)
                 if start_pos is None: raise RuntimeError("Failed to place human randomly")
                 self._human_pos = start_pos
            else:
                 self._human_pos = self.human_start_pos
                 self.put_obj(HumanNPC(), self._human_pos[0], self._human_pos[1])
        else:
            # Place randomly, avoiding the goal
            #start_pos = self.place_obj(HumanNPC(), reject_pos=[self._human_goal_pos], max_tries=100)
            start_pos = self.place_obj(HumanNPC(), max_tries=100)
            if start_pos is None: raise RuntimeError("Failed to place human randomly")
            self._human_pos = start_pos
        # print(f"Placed human at: {self._human_pos}")


        # 3. Place the agent
        if self.agent_start_pos is not None:
             # Check if agent start conflicts with goal or human
             if self.agent_start_pos == self._human_goal_pos or self.agent_start_pos == self._human_pos:
                 print(f"Warning: Agent start pos {self.agent_start_pos} occupied by goal or human. Placing randomly.")
                 self.place_agent(max_tries=100)
             else:
                 self.agent_pos = self.agent_start_pos
                 self.agent_dir = 0 # Start facing right (consistent)
                 # Ensure the cell is marked as empty for agent rendering logic?
                 # self.grid.set(self.agent_pos[0], self.agent_pos[1], None) # Might not be needed
        else:
             # Place randomly, avoiding goal and human
             self.place_agent(max_tries=100)

        # print(f"Placed agent at: {self.agent_pos}")
        self.mission = "Help the human reach the green goal square"


    def gen_obs(self):
        # Return the dictionary observation based on current state
        agent_pos = tuple(self.agent_pos) if self.agent_pos is not None else (-1, -1)
        human_pos = tuple(self._human_pos) if self._human_pos is not None else (-1, -1)
        goal_pos = tuple(self._human_goal_pos) if self._human_goal_pos is not None else (-1, -1)

        obs = {
            "agent_pos": agent_pos,
            "human_pos": human_pos,
            "goal_pos": goal_pos
        }
        # print(f"Generated obs: {obs}")
        return obs

    def reset(self, *, seed=None, options=None):
        # Call parent reset to set up grid, agent position etc.
        super().reset(seed=seed) # This calls _gen_grid

        # Generate the first Dict observation
        obs = self.gen_obs()

        # Get the custom info (human/goal pos)
        info = self._get_info()

        # Reset step count
        self.step_count = 0

        # print(f"Reset done. Obs: {obs}, Info: {info}")
        return obs, info

    def _get_info(self):
         # Provide human and goal positions in info dict for easy access by agent
         human_pos = tuple(self._human_pos) if self._human_pos is not None else (-1, -1)
         goal_pos = tuple(self._human_goal_pos) if self._human_goal_pos is not None else (-1, -1)
         return {"human_pos": human_pos, "goal_pos": goal_pos}

    def _move_human_npc(self):
        """Moves the human NPC one step towards its goal. Returns True if moved."""
        if self._human_pos is None or self._human_goal_pos is None or self._human_pos == self._human_goal_pos:
            return False # Cannot or no need to move

        hx, hy = self._human_pos
        gx, gy = self._human_goal_pos
        original_pos = self._human_pos

        # Simple greedy movement towards goal
        dx = np.sign(gx - hx)
        dy = np.sign(gy - hy)

        moved = False
        # Try moving horizontally first if primary direction or only option
        if dx != 0 and abs(gx - hx) >= abs(gy - hy):
            next_pos_x = (hx + dx, hy)
            cell_x = self.grid.get(next_pos_x[0], next_pos_x[1])
            # Can move if cell is empty, the goal, or overlaps (e.g., agent)
            # Cannot move into walls
            if cell_x is None or cell_x.type == 'goal' or cell_x.can_overlap():
                 self._human_pos = next_pos_x
                 moved = True

        # Try moving vertically if horizontal failed or is primary direction
        if not moved and dy != 0:
            next_pos_y = (hx, hy + dy)
            cell_y = self.grid.get(next_pos_y[0], next_pos_y[1])
            if cell_y is None or cell_y.type == 'goal' or cell_y.can_overlap():
                 self._human_pos = next_pos_y
                 moved = True

        # If primary direction failed, try the other direction (if not already tried)
        if not moved and dx != 0 and abs(gx - hx) < abs(gy - hy):
             next_pos_x = (hx + dx, hy)
             cell_x = self.grid.get(next_pos_x[0], next_pos_x[1])
             if cell_x is None or cell_x.type == 'goal' or cell_x.can_overlap():
                 self._human_pos = next_pos_x
                 moved = True

        # print(f"Human move attempt: {original_pos} -> {self._human_pos} (Goal: {self._human_goal_pos})")
        return moved


    def step(self, action):
        """Agent takes action, then Human NPC moves. Returns Dict obs."""
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False

        # --- 1. Agent Action ---
        # Store previous agent pos/dir
        prev_agent_pos = self.agent_pos
        prev_agent_dir = self.agent_dir

        # Update agent state based on action
        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
        # Move forward
        elif action == self.actions.forward:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            # Allow moving into empty, goal, or overlapping cells (like human)
            if fwd_cell is None or fwd_cell.can_overlap() or fwd_cell.type == 'goal':
                self.agent_pos = tuple(fwd_pos)
            # Handle pickup, drop, toggle if needed - currently ignored by agent map
        # elif action == self.actions.pickup: ...
        # elif action == self.actions.drop: ...
        # elif action == self.actions.toggle: ...
        elif action == self.actions.done: # 'done' action exists but not used here
            pass
        else:
            # This case should ideally be prevented by the action mapping in main.py
            print(f"Warning: Unknown action {action} received in env.step. Agent stays.")

        # --- 2. Human NPC Movement ---
        # Clear human's old position on grid *before* moving
        # Only clear if it wasn't the goal cell
        if self._human_pos:
            cell_at_human = self.grid.get(self._human_pos[0], self._human_pos[1])
            if cell_at_human is None or cell_at_human.type != 'goal':
                 self.grid.set(self._human_pos[0], self._human_pos[1], None)

        # Move human
        human_moved = self._move_human_npc()

        # Place human object in new position on grid *after* moving
        # Only place if the new position isn't the goal cell
        if self._human_pos:
            cell_at_new_human = self.grid.get(self._human_pos[0], self._human_pos[1])
            if cell_at_new_human is None or cell_at_new_human.type != 'goal':
                 self.grid.set(self._human_pos[0], self._human_pos[1], HumanNPC())


        # --- 3. Calculate Rewards and Termination ---
        human_succeeded = False
        if self._human_pos == self._human_goal_pos:
            human_succeeded = True
            terminated = True # End episode if human reaches goal

        # Robot Reward (Proxy for Power): Reward agent if human succeeded
        if human_succeeded:
            reward = 1.0 # Positive reward for agent
        else:
            # Small step penalty for agent to encourage efficiency
            reward = -0.01

        # Base Human Reward (for agent's internal model update)
        # 1.0 if human reached goal *this step*, 0 otherwise
        r_h_obs = 1.0 if human_succeeded else 0.0

        # --- 4. Check Truncation ---
        if self.step_count >= self.max_steps:
            truncated = True

        # --- 5. Generate Observation and Info ---
        obs = self.gen_obs()
        info = self._get_info()

        # --- 6. Render ---
        if self.render_mode == "human":
            self.render()

        # print(f"Step {self.step_count}: Action={action}, AgentPos={obs['agent_pos']}, HumanPos={obs['human_pos']}, Reward={reward}, Term={terminated}, Trunc={truncated}")
        return obs, reward, r_h_obs, terminated, truncated, info


# Example usage (remains the same)
if __name__ == "__main__":
    env = PowerGridEnv(size=8, render_mode="human") # Use human render mode
    obs, info = env.reset()
    # env.render() # Render is called in reset if mode is human

    for i in range(50): # Max 50 steps example
        # Sample a valid environment action (0-6, but we only mapped 0-3 in main)
        # For testing, let's sample agent actions 0-3 and map them
        agent_action = env.action_space.sample() # Samples from the Dict space - WRONG!
        # Need to sample from the *agent's* action space (0-3)
        # Let's map env actions for testing here: 0:right, 1:down, 2:left, 3:up
        env_action = random.choice([0, 1, 2, 3]) # Sample a movement action

        print(f"Step {i+1}, Env Action: {env_action}")
        obs, reward, r_h_obs, terminated, truncated, info = env.step(env_action)
        # env.render() # Render is called in step if mode is human
        print(f"Agent Reward: {reward:.2f}, Human Base Reward: {r_h_obs}, Term: {terminated}, Trunc: {truncated}")
        print(f"Agent Pos: {obs['agent_pos']}, Human Pos: {info['human_pos']}, Goal Pos: {info['goal_pos']}")

        if terminated or truncated:
            print("Episode finished!")
            obs, info = env.reset()
            # env.render() # Render is called in reset if mode is human
            # break # Uncomment to stop after one episode

    env.close()
