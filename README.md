# IQL Power-Seeking Agent in a MiniGrid Environment

## Overview

This project implements a Reinforcement Learning (RL) agent designed to operate within a `minigrid` environment. The environment contains the controllable agent (a "Robot") and a non-player character (an "Human" NPC).

The primary goal is **not** simply to make the Robot help the Human reach a specific, hardcoded goal. Instead, the Robot learns to maximize a proxy measure of the Human's **"power"**. Power, in this context, represents the Human's *potential* ability to achieve a *variety* of possible goals within the environment.

The Robot learns this behavior using a modified Independent Q-Learning (IQL) algorithm, corresponding to Algorithm 1 in the source research ("Ram and Jobst: Power"). This specific algorithm uses a simplified model of the Human compared to the full theory presented in the paper.

## Core Concepts

This project introduces several key concepts beyond a standard RL setup:

1.  **Robot Agent:** The learning entity controlled by the IQL algorithm (`iql_agent.py`). It explores the environment and learns a policy.
2.  **Human NPC:** A character simulated within the environment (`minigrid_power_env.py`). It follows a simple, predefined behavior (e.g., moving towards a specific target location for that episode). **Crucially, the Robot does not have direct access to the NPC's internal logic.**
3.  **Potential Goals ($\mathcal{G}$):** This is a predefined set of states (e.g., grid coordinates) that the *Robot believes* the Human *might potentially* want to achieve. This set is defined by the engineer setting up the experiment and is provided to the Robot agent. It represents the space of possibilities the Robot considers when evaluating the Human's power.
4.  **Goal Prior ($\mu_g$):** This is the Robot's *belief* about the likelihood of each potential goal in $\mathcal{G}$ being the Human's *true* goal at any given time. It's a probability distribution over the potential goals, also defined by the engineer. For example, a uniform prior means the Robot assumes all potential goals are equally likely.
5.  **Robot's Internal Human Model ($Q_h, V_h$):** The Robot agent learns an internal model of the Human's behavior. This model consists of a Q-table, $Q_h(\text{state}, g, a_h)$, which estimates the expected future *base* rewards (like reaching a goal location) for the Human, *if* the Human were pursuing potential goal $g$ from the current state and took action $a_h$.
    * The Robot updates this $Q_h$ table based on the *actual base rewards* ($r_h^{obs}$) the Human NPC receives in the environment (e.g., +1 for reaching its target).
    * The Robot *assumes* the Human acts rationally (specifically, softmax-rationally with temperature $\beta_h$) with respect to these learned $Q_h$ values for whichever goal the Robot currently thinks the Human has.
    * From $Q_h$, the Robot can calculate the estimated state value $V_h(\text{state}, g)$ for the Human for each potential goal $g$.
    * **Important:** This learned model ($Q_h$) is the Robot's *prediction* of human behavior and value, distinct from the simple hardcoded logic of the Human NPC in the environment.
6.  **Robot's Internal Reward ($r_r^{calc}$):** This is the core of the power-seeking mechanism. Instead of using the reward signal directly from the environment step (which might just signal task completion), the Robot calculates its *own* reward signal based on how the last action affected the Human's estimated power in the resulting state ($s'$). The calculation is:
    * **a.** For the state $s'$ reached after the Robot's action, calculate the estimated Human value $\hat{V}_h(s', g')$ for *every potential goal* $g'$ in the set $\mathcal{G}$, using the internal $Q_h$ model.
    * **b.** Calculate an intermediate value $z(s')$ representing the expected *powered* value across the goal distribution:
        $z(s') = \sum_{g' \in \mathcal{G}} \mu_g(g') [\max(0, \hat{V}_h(s', g'))]^{1+\eta}$
        * $\mu_g(g')$ is the prior probability of potential goal $g'$.
        * $\eta \ge 0$ is a parameter weighting certainty. $\eta=0$ means the Robot values expected $V_h$. $\eta > 0$ means the Robot prefers situations where $V_h$ is high for goals the Human is likely to achieve (higher $V_h$ values contribute more than linearly).
    * **c.** Calculate the final internal reward using a concave, increasing function $f$:
        $r_r^{calc}(s') = f(z(s'))$
        * The function $f$ encourages distributing power/potential rather than concentrating it. A common choice is $f(z) = 2 - 2/(z + \epsilon)$ (where $\epsilon$ is small for stability), which is bounded between -inf and 2. Another is $f(z) = \log_2(z+\epsilon)$. Using $f(z)=z$ reduces this to maximizing the expected powered value.
7.  **Robot Learning ($Q_r$):** The Robot learns its own policy using a standard Q-table, $Q_r(\text{state}, a_r)$. It uses the internally calculated $r_r^{calc}$ as the reward signal in its Q-learning update rule:
    * $Q_r(s, a_r) \leftarrow Q_r(s, a_r) + \alpha_r [ (r_r^{calc}(s') + \gamma_r \max_{a'} Q_r(s', a')) - Q_r(s, a_r) ]$
8.  **Two Timescales:** The learning rate for the internal human model ($\alpha_h$) is typically set higher than the learning rate for the robot's own policy ($\alpha_r$). This allows the human model to adapt more quickly to the consequences of the robot's actions.
9.  **State Representation:** For these tabular implementations, the state used as keys in the Q-tables is derived from the MiniGrid observation, typically simplified to `(agent_pos_tuple, human_pos_tuple)`. Agent direction is ignored here for simplicity but could be added.

## Code Structure

* **`minigrid_power_env.py`**:
    * Defines `PowerGridEnv`, inheriting from `minigrid.minigrid_env.MiniGridEnv`.
    * Places the agent (Robot), an NPC (`HumanNPC` object, rendered as a ball), and a `Goal` object.
    * The `step` method first executes the agent's action using the parent `step`, then calculates and executes the Human NPC's move towards its goal.
    * Returns standard MiniGrid `obs`, the agent's *proxy reward* (e.g., +1 if human succeeded, else step penalty), the *human's base observed reward* $r_h^{obs}$ (e.g., +1 if human reached goal *this step*, 0 otherwise), `terminated`, `truncated`, and `info` (containing human/goal positions).
* **`iql_agent.py`**:
    * Defines `IQLPowerAgent`.
    * Initializes with action space size, the potential `goal_set` (list of coordinates), `goal_prior` (dict mapping goal index to probability), learning rates, etc.
    * Stores $Q_h$ and $Q_r$ as `defaultdict` mapping `state_tuple` -> numpy arrays.
    * `choose_robot_action`: Implements $\epsilon$-greedy based on $Q_r$.
    * `_get_state_tuple`: Converts positions to hashable tuple keys.
    * `_get_human_policy_for_goal`, `_calculate_V_h`: Compute the policy and value based on $Q_h$ for a given potential goal.
    * `update`: Performs the Q-learning updates for both $Q_h$ (using $r_h^{obs}$) and $Q_r$ (using internally calculated $r_r^{calc}$).
    * `get_human_action_for_simulation`: Samples an action from the learned human policy for a given goal (used within the `update` method for the $Q_h$ update).
* **`main.py`**:
    * Sets up the `PowerGridEnv` and `IQLPowerAgent` with desired parameters (grid size, potential goals, goal prior, learning rates, $\eta$, $f$, etc.).
    * Runs the main training loop:
        * Resets the environment.
        * Determines the actual goal index for the human NPC in this episode.
        * In each step: gets agent action, steps environment, gets next state info, simulates human action based on $Q_h$, calls agent's `update` method.
    * Logs rewards and success rates.
    * Optionally plots results.

## Dependencies

* `numpy`
* `minigrid`
* `matplotlib` (optional, for plotting in `main.py`)

Install dependencies using pip:
```bash
pip install numpy minigrid matplotlib
How to RunInstall the dependencies listed above.Place the files (minigrid_power_env.py, iql_agent.py, main.py) in the same directory.Configure parameters in main.py:Environment settings (env_config).The Robot's set of considered potential goals (potential_goals).The Robot's goal prior (goal_prior).Agent hyperparameters (learning rates alpha_h, alpha_r, discount factors gamma_h, gamma_r, human rationality beta_h, exploration epsilon_r, power parameters eta, f_func).Training loop settings (num_episodes).Run the main script from your terminal:python main.py
Notes & Potential ExtensionsState Space: The current tabular state (agent_pos, human_pos) is simple but grows quadratically with grid size. It ignores agent direction and object states. For more complex environments, function approximation (Deep Q-Networks) would be needed instead of tabular Q-learning.Human NPC: The current NPC is deterministic and simple. A more complex or stochastic NPC could be used. Remember the agent only observes the NPC's effect via state changes and rhobs​, it doesn't know the NPC's internal logic.Robot Reward: The internal reward rrcalc​ drives the Robot's behavior. Tuning the potential goal set G, prior μg​, function f, and parameter η is crucial for achieving desired emergent behaviors.Algorithm 2: This codebase implements Algorithm 1. Implementing Algorithm 2 (two-phase learning with the cautious Qm​ human model) would require significant changes to the agent logic, including storing Qm​ and implementing the two distinct learning phases.MiniGrid Actions: The current agent maps its internal actions (0-3) to MiniGrid movement actions. It could be extended to use other actions like toggle, pickup, `