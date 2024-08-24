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
from utils.visualization_routines import *
from RAN_simulator import RAN_Simulator
import numpy as np
import pandas as pd
import random

# Create RAN simulator object
num_cells = 2
simulator_instance = RAN_Simulator(num_cells)

# Set the agent model
agent = DQN.load('Agent_models/two_cells/1000_step/3k_ep/0_3_fraction.zip')

# Define initial variables
num_prbs_to_allocate = [250, 330]
starting_point = [120, 90]

"""
#Example four cels
# Set the agent model
agent = DQN.load('Agent_models/four_cells/1000_step/3k_ep/0_3_fraction.zip')

# Define initial variables
num_prbs_to_allocate = [250, 120, 50, 280]
starting_point = [230, 340, 420, 50]
"""

# Generate UE position randomized
if num_cells == 2:
    scenario_config_file = 'config_files/scenario_config_2_cells.csv'
else:
    scenario_config_file = 'config_files/scenario_config_4_cells.csv'

scenario_config_df = pd.read_csv(scenario_config_file, sep=';')
ues_pos = np.zeros((2, scenario_config_df['num_ues'].iloc[0]))

for i in range(scenario_config_df['num_ues'].iloc[0]):
    ues_pos[0, i] = random.randint(0, scenario_config_df['scenario_x_size'].iloc[0])
    ues_pos[1, i] = random.randint(0, scenario_config_df['scenario_y_size'].iloc[0])

# Start RAN simulator object
obs, initial_pos, final_pos, interference, thr_cells, serving_BS, thr_UE_ID = simulator_instance.start(num_prbs_to_allocate, starting_point, ues_pos)

# Plot scenario
plot_scenario(num_cells, scenario_config_df['num_ues'].iloc[0], ues_pos[0,:], ues_pos[1,:], serving_BS, thr_cells, thr_UE_ID)

# Store observation to results visualization
actions_array = np.zeros((num_cells, 1))
thr = thr_cells[:, np.newaxis]
prbs_assigned = obs[:, 1][:, np.newaxis]
prbs_interference = interference[:, np.newaxis]
prbs_initial_pos = initial_pos[:, np.newaxis]
prbs_final_pos = final_pos[:, np.newaxis]
done = False

# Plot Initial Bandwidth allocation
plot_prbs_array(num_cells, prbs_initial_pos[:, 0], prbs_final_pos[:, 0], "Initial bandwidth allocation", num_prbs_to_allocate)

max_steps = 500
iter = 0
while done == False:
    actions = []
    for i in range(num_cells):
        actions.append(agent.predict(obs[i], deterministic=True))

    # Convert action to numpy array
    actions = [tupla[0] for tupla in actions]

    if all(x == 6 for x in actions) or iter == max_steps:
        done = True
    else:
        # Introduce actions into the simulator
        obs, initial_pos, final_pos, interference, thr_cells = simulator_instance.run(actions)
        thr = np.hstack((thr, thr_cells[:, np.newaxis]))
        prbs_assigned = np.hstack((prbs_assigned, obs[:, 1][:, np.newaxis]))
        prbs_interference = np.hstack((prbs_interference, interference[:, np.newaxis]))
        prbs_initial_pos = np.hstack((prbs_initial_pos, initial_pos[:, np.newaxis]))
        prbs_final_pos = np.hstack((prbs_final_pos, final_pos[:, np.newaxis]))
        actions_array = np.hstack((actions_array, np.array(actions).reshape(-1, 1)))
    iter += 1

# Plot Final Bandwidth allocation
plot_prbs_array(num_cells, prbs_initial_pos[:, -1], prbs_final_pos[:, -1], "Final bandwidth allocation", prbs_assigned[:,-1])

# Plot Throughput evolution step by step
plot_thr(thr, actions_array)

# Plot number of PRBs assigned evolution step by step
plot_assigned_prbs(prbs_assigned, actions_array)

# Plot number of PRBs interfered evolution step by step
plot_interference_prbs(prbs_interference, actions_array)
