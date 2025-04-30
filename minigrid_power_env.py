import numpy as np

from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX, IDX_TO_OBJECT
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, WorldObj, Ball # Using Ball to represent Human
from minigrid.minigrid_env import MiniGridEnv

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
        # Render as a blue ball (or choose another representation)
        fill_coords(img, point_in_rect(0.25, 0.75, 0.25, 0.75), COLORS[self.color])

# Import rendering helpers after defining HumanNPC if needed
from minigrid.utils.rendering import fill_coords, point_in_rect
from minigrid.core.constants import COLORS


class PowerGridEnv(MiniGridEnv):
    """
    MiniGrid environment with a Robot (agent) and a Human (NPC).
    The agent's goal is implicitly to help the human reach its goal.
    """
    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        human_start_pos=None, # If None, random placement
        human_goal_pos=None,  # If None, random placement
        max_steps=None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.human_start_pos = human_start_pos
        self.human_goal_pos = human_goal_pos

        # Store human position internally
        self._human_pos = None
        self._human_goal_pos = None # Store the actual goal position for the episode

        mission_space = MissionSpace(mission_func=lambda: "Help the human reach the goal")

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True, # Agent sees everything for tabular state
            agent_view_size=size,   # Agent sees the whole grid
            **kwargs,
        )
        # Observation space needs to include human position if not using image obs
        # For tabular, we'll extract info, but let's define a reasonable space
        # If using agent_pos, agent_dir, human_pos, goal_pos:
        # self.observation_space = spaces.Dict({
        #     "agent_pos": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #     "agent_dir": spaces.Discrete(4),
        #     "human_pos": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #     "goal_pos": spaces.Box(0, size - 1, shape=(2,), dtype=int)
        # })
        # Using the default image observation space for now, agent extracts info
        # print(f"Initial observation space: {self.observation_space}")


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the grid walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the human goal
        if self.human_goal_pos is not None:
            goal = Goal()
            self.put_obj(goal, self.human_goal_pos[0], self.human_goal_pos[1])
            self._human_goal_pos = self.human_goal_pos
        else:
            self._human_goal_pos = self.place_obj(Goal())

        # Place the human NPC
        if self.human_start_pos is not None:
            self._human_pos = self.human_start_pos
            # Check if the cell is empty before placing
            if self.grid.get(self._human_pos[0], self._human_pos[1]) is None:
                 self.grid.set(self._human_pos[0], self._human_pos[1], HumanNPC())
            else: # Fallback if start pos is occupied (e.g., by goal)
                 print(f"Warning: Human start pos {self.human_start_pos} occupied, placing randomly.")
                 self._human_pos = self.place_obj(HumanNPC(), max_tries=100)
        else:
             self._human_pos = self.place_obj(HumanNPC(), max_tries=100)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = 0 # Start facing right
            # Check if the cell is empty before placing
            if self.grid.get(self.agent_pos[0], self.agent_pos[1]) is None:
                 self.grid.set(self.agent_pos[0], self.agent_pos[1], None) # Ensure start is clear for agent rendering
            else:
                 print(f"Warning: Agent start pos {self.agent_start_pos} occupied, placing randomly.")
                 self.place_agent() # Random placement if start is bad
        else:
            self.place_agent()

        self.mission = "Help the human reach the green goal square"


    def _get_obs(self):
        # Standard image observation (agent will extract info)
        # If using Dict space, populate it here:
        # obs = {
        #     "agent_pos": self.agent_pos,
        #     "agent_dir": self.agent_dir,
        #     "human_pos": self._human_pos,
        #     "goal_pos": self._human_goal_pos
        # }
        # return obs
        return super()._get_obs() # Returns image observation by default

    def _get_info(self):
         # Provide human and goal positions in info dict for easy access by agent
         return {"human_pos": self._human_pos, "goal_pos": self._human_goal_pos}

    def _move_human_npc(self):
        """Moves the human NPC one step towards its goal."""
        if self._human_pos is None or self._human_goal_pos is None:
            return

        hx, hy = self._human_pos
        gx, gy = self._human_goal_pos

        # Simple greedy movement towards goal
        dx = np.sign(gx - hx)
        dy = np.sign(gy - hy)

        # Try moving horizontally first if primary direction
        if abs(gx - hx) >= abs(gy - hy):
            next_pos_x = (hx + dx, hy)
            cell_x = self.grid.get(next_pos_x[0], next_pos_x[1])
            if dx != 0 and (cell_x is None or cell_x.type == 'goal' or cell_x.can_overlap()):
                 self._human_pos = next_pos_x
                 return # Moved horizontally

            # Try vertically if horizontal failed or not primary
            next_pos_y = (hx, hy + dy)
            cell_y = self.grid.get(next_pos_y[0], next_pos_y[1])
            if dy != 0 and (cell_y is None or cell_y.type == 'goal' or cell_y.can_overlap()):
                 self._human_pos = next_pos_y
                 return # Moved vertically
        else: # Try moving vertically first if primary direction
            next_pos_y = (hx, hy + dy)
            cell_y = self.grid.get(next_pos_y[0], next_pos_y[1])
            if dy != 0 and (cell_y is None or cell_y.type == 'goal' or cell_y.can_overlap()):
                 self._human_pos = next_pos_y
                 return # Moved vertically

            # Try horizontally if vertical failed or not primary
            next_pos_x = (hx + dx, hy)
            cell_x = self.grid.get(next_pos_x[0], next_pos_x[1])
            if dx != 0 and (cell_x is None or cell_x.type == 'goal' or cell_x.can_overlap()):
                 self._human_pos = next_pos_x
                 return # Moved horizontally

        # If cannot move closer, stay put (or add random move?)
        # Staying put for simplicity

    def step(self, action):
        """Agent takes action, then Human NPC moves."""
        # Agent action first
        obs, reward, terminated, truncated, info = super().step(action)

        # Now move the human NPC *after* the agent has acted
        # Clear human's old position on grid for rendering
        if self._human_pos:
            self.grid.set(self._human_pos[0], self._human_pos[1], None)

        # Move human
        self._move_human_npc()

        # Place human in new position on grid for rendering
        if self._human_pos:
            # If human reached goal, place goal back, otherwise human
            obj_at_new_pos = self.grid.get(self._human_pos[0], self._human_pos[1])
            if not (obj_at_new_pos and obj_at_new_pos.type == 'goal'):
                 self.grid.set(self._human_pos[0], self._human_pos[1], HumanNPC())


        # --- Calculate Rewards and Termination based on HUMAN state ---
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

        # Update observation and info after human move
        obs = self._get_obs()
        info = self._get_info() # Ensure info has updated human pos

        # Check max steps termination
        if self.step_count >= self.max_steps:
            truncated = True

        return obs, reward, r_h_obs, terminated, truncated, info

# Example usage:
if __name__ == "__main__":
    env = PowerGridEnv(size=8, render_mode="human") # Use human render mode
    obs, info = env.reset()
    env.render()

    for i in range(50): # Max 50 steps example
        action = env.action_space.sample() # Sample random action
        print(f"Step {i+1}, Agent Action: {action}")
        obs, reward, r_h_obs, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Agent Reward: {reward:.2f}, Human Base Reward: {r_h_obs}, Term: {terminated}, Trunc: {truncated}")
        print(f"Agent Pos: {env.agent_pos}, Human Pos: {info['human_pos']}, Goal Pos: {info['goal_pos']}")

        if terminated or truncated:
            print("Episode finished!")
            obs, info = env.reset()
            env.render()
            # break # Uncomment to stop after one episode

    env.close()

