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
from Environment import Environment
import pandas as pd
import random
import os
import numpy as np

def random_starting_point(num_PRBs):
    """
    This routine generates an initial position for the random allocation of PRBs.
    The positions must be multiples of 10.

    Inputs:
        num_PRBs (int): Number of PRBs to assign.
    Outpus:
        starting_point (int). Initial position to assignation.
    """
    num_random = np.random.randint(0, 556 - num_PRBs)
    if num_random == 0:
        return num_random
    elif num_random % 10 == 0:
        return num_random
    else:
        return num_random - (num_random % 10)


# Setting up the environments
num_cells = 2
num_steps = 1000

# List of model files to load

model_files = ['two_cells_models/500_step/1k_ep/0_95_gamma_0_3_fraction.zip',
               'two_cells_models/500_step/1k_ep/0_95_gamma_0_7_fraction.zip',
               'two_cells_models/500_step/2k_ep/0_95_gamma_0_3_fraction.zip',
               'two_cells_models/500_step/2k_ep/0_95_gamma_0_7_fraction.zip',
               'two_cells_models/500_step/3k_ep/0_95_gamma_0_3_fraction.zip',
               'two_cells_models/500_step/3k_ep/0_95_gamma_0_7_fraction.zip',
               'two_cells_models/500_step/4k_ep/0_95_gamma_0_3_fraction.zip',
               'two_cells_models/500_step/4k_ep/0_95_gamma_0_7_fraction.zip',
               'two_cells_models/700_step/1k_ep/0_95_gamma_0_3_fraction.zip',
               'two_cells_models/700_step/1k_ep/0_95_gamma_0_7_fraction.zip',
               'two_cells_models/700_step/2k_ep/0_95_gamma_0_3_fraction.zip',
               'two_cells_models/700_step/2k_ep/0_95_gamma_0_7_fraction.zip',
               'two_cells_models/700_step/3k_ep/0_95_gamma_0_3_fraction.zip',
               'two_cells_models/700_step/3k_ep/0_95_gamma_0_7_fraction.zip',
               'two_cells_models/700_step/4k_ep/0_95_gamma_0_3_fraction.zip',
               'two_cells_models/700_step/4k_ep/0_95_gamma_0_7_fraction.zip',
               'two_cells_models/1000_step/1k_ep/0_95_gamma_0_3_fraction.zip',
               'two_cells_models/1000_step/1k_ep/0_95_gamma_0_7_fraction.zip',
               'two_cells_models/1000_step/2k_ep/0_95_gamma_0_3_fraction.zip',
               'two_cells_models/1000_step/2k_ep/0_95_gamma_0_7_fraction.zip',
               'two_cells_models/1000_step/3k_ep/0_95_gamma_0_3_fraction.zip',
               'two_cells_models/1000_step/3k_ep/0_95_gamma_0_7_fraction.zip',
               'two_cells_models/1000_step/4k_ep/0_95_gamma_0_3_fraction.zip',
               'two_cells_models/1000_step/4k_ep/0_95_gamma_0_7_fraction.zip'
               ]


# Create environments and load models
num_models = len(model_files)
envs = [Environment(num_cells, num_steps, False) for _ in range(num_models)]
models = [DQN.load(model_file) for model_file in model_files]

# Set the number of episodes during inference
num_episodes = 100

# Load scenario configuration files
scenario_config_file = 'config_files/scenario_config_2_cells.csv' if num_cells == 2 else 'config_files/scenario_config_4_cells.csv'
scenario_config_df = pd.read_csv(scenario_config_file, sep=';')
num_ues = scenario_config_df['num_ues'].iloc[0]
scenario_x_size = scenario_config_df['scenario_x_size'].iloc[0]
scenario_y_size = scenario_config_df['scenario_y_size'].iloc[0]
gnb_id = list(range(num_cells))

# Variables to store initial values common to all models
thr_ini = []
PRBs_ini = []
PRBs_interf_ini = []

# Variables to store final values for each model
thr_end = [[] for _ in range(num_models)]
PRBs_end = [[] for _ in range(num_models)]
PRBs_interf_end = [[] for _ in range(num_models)]
reward_per_episode = [[] for _ in range(num_models)]

# List of possible initial PRBs, modify other values if desired
random_prbs_list = [10, 20, 30, 40, 50, 60, 70, 90, 110, 130, 150, 170, 190, 210, 250, 270, 300, 330, 370, 400, 430, 450, 500]
for i in range(num_episodes):
    # Randomly select the cell controlled by the agent
    agent_id = random.choice(gnb_id)

    # Generate random PRBs and position
    num_PRBs_to_allocate = [random.choice(random_prbs_list) for _ in range(num_cells)]
    starting_point = [random_starting_point(prb) for prb in num_PRBs_to_allocate]

    # Generate random UEs position
    x = np.random.randint(0, scenario_x_size + 1, num_ues)
    y = np.random.randint(0, scenario_y_size + 1, num_ues)

    # Reset environments and get initial observations
    obs = []
    for env in envs:
        ob, prbs_interf, thr = env.manual_reset(num_PRBs_to_allocate, starting_point, agent_id, x, y)
        obs.append((ob, prbs_interf, thr))

    # Save initial states
    thr_ini.append(obs[0][2])
    PRBs_ini.append(obs[0][0][1])
    PRBs_interf_ini.append(obs[0][1])

    done = [False] * num_models
    rewards_accumulated = [0] * num_models
    steps = [0] * num_models
    # While all the agents have not finished the episode continues.
    while not all(done):
        for i in range(num_models):
            if not done[i]:
                action, _ = models[i].predict(obs[i][0], deterministic=True)
                ob, reward, done[i], info, prbs_interf, thr = envs[i].step(action)
                rewards_accumulated[i] += reward
                steps[i] += 1
                obs[i] = (ob, prbs_interf, thr)

                if action == 6:  # Specific action that ends the episode
                    done[i] = True

    for i in range(num_models):
        reward_per_episode[i].append(rewards_accumulated[i])
        thr_end[i].append(obs[i][2])
        PRBs_end[i].append(obs[i][0][1])
        PRBs_interf_end[i].append(obs[i][1])

# Store results in csv and print key metrics
os.makedirs('inference_results', exist_ok=True)
print("Summary of results:")
results = []
for i in range(num_models):
    print("\nModel {}: {}".format(i, model_files[i]))
    results = {
        'thr_ini': thr_ini,
        'prbs_ini': PRBs_ini,
        'interf_ini': PRBs_interf_ini,
        'thr_end': thr_end[i],
        'prbs_end': PRBs_end[i],
        'interf_end': PRBs_interf_end[i],
        'reward': reward_per_episode[i]
    }
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'inference_results/results_model_{i}.csv', index=False)

    # 1) Success Rate (%)
    total_episodios = len(df_results)
    total_ep_success = len(df_results[df_results['thr_end'] >= scenario_config_df['thr_min'].iloc[0]])
    success_Rate = (total_ep_success / total_episodios) * 100
    print(" - Success rate : {:.2f}%".format(success_Rate))

    # 2)  Average throughput (Mbps)
    filtered_df = df_results[df_results['thr_end'] >= scenario_config_df['thr_min'].iloc[0]]
    mean_thr_end = filtered_df['thr_end'].mean()
    print(" - Average throughput (Mbps): {:.2f}Mbps".format(mean_thr_end))

    # 3) Interference-free rate (%)
    interf_free_rate = ( (df_results['interf_end'] == 0.0).sum() / len(df_results)) * 100

    print(" - Interference-free rate {:.2f}%".format(interf_free_rate))
