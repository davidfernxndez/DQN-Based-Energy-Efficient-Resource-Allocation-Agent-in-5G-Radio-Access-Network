# Coded by: David Fernández Martínez (davidfm8@correo.ugr.es)
# last modification: 24/August/2024
#
# Copyright (C) 2024 David Fernández Martínez. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from Environment import Environment

# Set up the environment
num_cells = 2
num_steps = 100

# Set the number of episodes during training
num_episodes = 3

# Set the exploration fraction value
exploration_fraction = 0.3

# Initialize the environment with the specified parameters
env = Environment(num_cells, num_steps, True)

# Check if the custom environment adheres to the Gym interface
# If the environment doesn't follow the interface, an error will be thrown
check_env(env, warn=True)

# Wrap the environment to make it compatible with vectorized environments
env = make_vec_env(lambda: env, n_envs=1)

# Train the agent using the DQN algorithm
print("Training the agent")
model = DQN(
    'MlpPolicy',  # Policy type, using a multi-layer perceptron (MLP) neural network
    env,  # The environment in which the agent will be trained
    gamma=0.95,  # Discount factor for reward computation
    learning_rate=0.001,  # Rate at which the model learns
    learning_starts=500,  # Number of steps before the agent starts learning
    verbose=1,  # Verbosity level: 1 for progress updates
    exploration_fraction=exploration_fraction,  # Fraction of training steps where exploration is prioritized
    exploration_initial_eps=1.0,  # Initial value of epsilon in epsilon-greedy strategy (fully exploratory)
    exploration_final_eps=0.05,  # Final value of epsilon in epsilon-greedy strategy (less exploratory)
    tensorboard_log="./tensorboard"  # Directory to log training progress for TensorBoard visualization
)

# Start the learning process, training for a total of [max_steps * num_episodes] steps
model.learn(num_steps * num_episodes)
model.save(f"{num_cells}_cells_{num_steps}_steps_{num_episodes}_ep_{str(exploration_fraction).replace('.', '_')}_fraction")

