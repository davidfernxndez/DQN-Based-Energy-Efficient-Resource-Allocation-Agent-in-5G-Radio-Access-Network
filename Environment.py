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

import numpy as np
import pandas as pd
import gym
import math
import random
from gym import spaces


class Environment(gym.Env):
    """
    Custom Environment that follows gym interface.
    This environment implements a scenario where a 5G NG-RAN is deployed
    in which an agent can interact by modifying the PRBs of a cell in the scenario.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    # Define constants for clearer code

    def __init__(self, num_cells_scenario, max_steps, mode):
        """
        Initialize the environment.

        Parameters:
        num_cells_scenario (int): Identifier for the scenario configuration (2 or 4).
        max_steps (int): Maximum number of steps per episode.
        mode (str): Mode of operation: True (Train), False (Inference).
        """
        super(Environment, self).__init__()
        # Store operation mode
        self.mode = mode
        # Define action and observation space
        self.n_actions = 7
        self.action_space = spaces.Discrete(self.n_actions)

        # Define observation space and config files
        if num_cells_scenario == 2:
            scenario_config_file = 'config_files/scenario_config_2_cells.csv'
            simulator_config_file = 'config_files/simulator_config_2_cells.csv'
            self.observation_space = spaces.MultiDiscrete([62, 556, 4, 556, 3])
        elif num_cells_scenario == 4:
            scenario_config_file = 'config_files/scenario_config_4_cells.csv'
            simulator_config_file = 'config_files/simulator_config_4_cells.csv'
            self.observation_space = spaces.MultiDiscrete([62, 556, 4, 556, 3, 556, 3, 556, 3])

        # Load scenario configuration from CSV
        scenario_config_df = pd.read_csv(scenario_config_file, sep=';')

        # Number of PRBs modified by the agent
        self.agent_num_prbs = scenario_config_df['agent_num_prbs'].iloc[0]

        # Actions codification
        self.actions_mapping = {
            0: (-self.agent_num_prbs, -1, 0),
            1: (-self.agent_num_prbs, 1, 0),
            2: (self.agent_num_prbs, -1, 0),
            3: (self.agent_num_prbs, 1, 0),
            4: (0, 0, self.agent_num_prbs),
            5: (0, 0, -self.agent_num_prbs),
            6: (0, 0, 0),
        }
        # Set the maximum number of steps for an episode
        self.max_steps = max_steps

        # Scenario configuration variables
        self.num_cells = scenario_config_df['num_cells'].iloc[0]
        self.num_prbs = scenario_config_df['num_prbs'].iloc[0]
        self.bw_prb = scenario_config_df['bw_prb'].iloc[0]  # MHz
        self.thr_min = scenario_config_df['thr_min'].iloc[0]
        self.num_ues = scenario_config_df['num_ues'].iloc[0]
        self.scenario_x_size = scenario_config_df['scenario_x_size'].iloc[0]
        self.scenario_y_size = scenario_config_df['scenario_y_size'].iloc[0]
        self.px_size = scenario_config_df['px_size'].iloc[0]
        self.cell_x_pos = eval(scenario_config_df['cell_x_pos'].iloc[0])
        self.cell_y_pos = eval(scenario_config_df['cell_y_pos'].iloc[0])

        # Generate a int identifier for each cell
        self.gnb_id = list(range(self.num_cells))

        # Read the csv simulator configuration file
        simulator_config_df = pd.read_csv(simulator_config_file, sep=';')

        # Simulator configuration variables
        self.femto_height = simulator_config_df['femto_height'].iloc[0]
        self.users_height = simulator_config_df['users_height'].iloc[0]
        self.LOS = simulator_config_df['LOS'].iloc[0]
        self.NLOS = simulator_config_df['NLOS'].iloc[0]
        self.femto_k_NLOS = simulator_config_df['femto_k_NLOS'].iloc[0]
        self.femto_k_LOS = simulator_config_df['femto_k_LOS'].iloc[0]
        self.femto_alpha_NLOS = simulator_config_df['femto_alpha_NLOS'].iloc[0]
        self.femto_alpha_LOS = simulator_config_df['femto_alpha_LOS'].iloc[0]
        self.PIRE = simulator_config_df['PIRE'].iloc[0]
        self.power_per_prb = simulator_config_df['power_tx_mw'].iloc[0] / self.num_prbs  # mW
        self.br_embb = simulator_config_df['br_embb'].iloc[0]
        self.femto_avg_SE = simulator_config_df['femto_avg_SE'].iloc[0]
        self.ue_pnoise_ch_femto = simulator_config_df['ue_pnoise_ch_femto'].iloc[0]
        self.SINRmin = simulator_config_df['SINRmin'].iloc[0]
        self.SINRmax = simulator_config_df['SINRmax'].iloc[0]
        self.Smax = simulator_config_df['Smax'].iloc[0]
        self.alpha_thr = simulator_config_df['alpha_thr'].iloc[0]
        self.scheduling_SE_param = simulator_config_df['scheduling_SE_param'].iloc[0]
        self.ue_scheduling_strategy = simulator_config_df['ue_scheduling_strategy'].iloc[0]

    def reset(self):
        """
        Resets the environment to an initial state and generates a new observation.

        The positions of the UEs within the scenario's dimensions, the number of 
        PRBs allocated per cell, their spectral placement, and the cell managed 
        by the agent are all assigned randomly.
        
        Returns:
            obs (numpy array): Observation that describes the state of the cell managed by the agent.
        """
        # Begin a episode
        self.num_step = 0

        # Initialization bandwidth_allocation  as zeros numpy arrays
        self.bandwidth_allocation = np.zeros((self.num_cells, self.num_prbs))

        # Generate the Agent Cell id
        self.agent_id = random.choice(self.gnb_id)

        # Generate the list of neighbor Cell id
        self.nbr_list = [nbr_id for nbr_id in self.gnb_id if nbr_id != self.agent_id]

        # Generate initial random PRBs to Cells and starting_point
        random_prbs_list = [10, 20, 30, 40, 50, 60, 70, 90, 110, 130, 150, 170, 190, 210, 250, 270, 300, 330, 370, 400,
                            430, 450, 500]
        self.initial_PRB_pos = []
        self.final_PRB_pos = []
        for i in range(self.num_cells):
            num_PRBs_to_allocate = random.choice(random_prbs_list)
            starting_point = self.random_starting_point(num_PRBs_to_allocate)
            # Allocate PRBs in bandwidth_allocation 
            self.bandwidth_allocation[i][starting_point:starting_point + num_PRBs_to_allocate] = 1
            # Extract initial and final pos for the assignation
            self.initial_PRB_pos.append(np.where(self.bandwidth_allocation[i] == 1)[0][0])
            self.final_PRB_pos.append(np.where(self.bandwidth_allocation[i] == 1)[0][-1])

        # Generate UEs position in scenario dimensions
        self.ues_x_pos = np.random.randint(0, self.scenario_x_size + 1, self.num_ues)
        self.ues_y_pos = np.random.randint(0, self.scenario_y_size + 1, self.num_ues)

        # Determine Serving cell for each UE
        _ = self.assigned_UE_to_cell()

        # Call the simulator
        thr_per_cell = self.simulator()

        # Extract throughput and PRBs assigned for agent cell
        self.thr = thr_per_cell[self.agent_id]
        self.assigned_PRBs = np.sum(self.bandwidth_allocation[self.agent_id])

        # Calculate PRB position flag
        self.prb_position_flag = self.calculate_prb_position_flag()

        # Obtain the number of interference PRBs in agent cell
        self.total_interference_PRBs = self.extract_total_PRBs_interference()

        if self.total_interference_PRBs == 0:
            # No interference
            self.nbr_interference_PRBs = (self.num_cells - 1) * [0]
            self.interference_pos_flag = (self.num_cells - 1) * [0]
        else:
            # Obtain the number of interference PRBs and interference flag from each neighbour cell
            self.nbr_interference_PRBs = []
            self.interference_pos_flag = []
            for nbr_id in self.nbr_list:
                num_interf_prbs = np.sum(
                    np.multiply(self.bandwidth_allocation[self.agent_id], self.bandwidth_allocation[nbr_id]))

                if num_interf_prbs == 0:
                    # No interference with this neighbour cell
                    self.nbr_interference_PRBs.append(num_interf_prbs)
                    self.interference_pos_flag.append(0)
                else:
                    # Interference with this neighbour cell
                    self.nbr_interference_PRBs.append(num_interf_prbs)
                    self.interference_pos_flag.append(self.calculate_interference_pos_flag(nbr_id))

        # Discretization of throughput
        self.thr_discrete = self.discretize_thr(self.thr)

        # Construct Observation array: [throughput, number of assigned PRBs, PRB position flag,
        # number of interference PRBs with each neighbour cell, interference PRB flag with each neighbour cell]
        if self.num_cells == 2:
            obs = np.array([self.thr_discrete, int(self.assigned_PRBs), self.prb_position_flag,
                            int(self.nbr_interference_PRBs[0]), self.interference_pos_flag[0]
                            ])
        elif self.num_cells == 4:
            obs = np.array([self.thr_discrete, int(self.assigned_PRBs), self.prb_position_flag,
                            int(self.nbr_interference_PRBs[0]), self.interference_pos_flag[0],
                            int(self.nbr_interference_PRBs[1]), self.interference_pos_flag[1],
                            int(self.nbr_interference_PRBs[2]), self.interference_pos_flag[2]
                            ])

        self.prev_action = None

        return obs

    def manual_reset(self, num_PRBs_to_allocate, starting_point, id, x_pos, y_pos):
        """
        Resets the environment to an initial state and generates a new observation.

        The positions of the UEs within the scenario dimensions, the number of PRBs
        assigned per cell, their spectral location and the cell managed by the agent
        are set manually through the input parameters.

        Parameters:
            num_PRBs_to_allocate (list): List with the PRBs to be assigned to each cell
            starting_point (list): List with the positions where the allocation of PRBs begins in each cell
            id (int): Integer that identifies the cell controlled by the agent.
            x_pos (numpy array): Numpy array with the positions of the UEs in the x dimension
            y_pos (numpy array): Numpy array with the positions of the UEs in the y dimension
        Returns:
            obs (numpy array): Observation that describes the state of the cell managed by the agent.
        """
        # Begin a episode
        self.num_step = 0

        # Initialization bandwidth_allocation  as zeros numpy arrays
        self.bandwidth_allocation = np.zeros((self.num_cells, self.num_prbs))

        # Set the Agent Cell id
        self.agent_id = id

        # Generate the list of neighbor Cell id
        self.nbr_list = [nbr_id for nbr_id in self.gnb_id if nbr_id != self.agent_id]

        # Assignment of PRBs in the initial positions
        self.initial_PRB_pos = []
        self.final_PRB_pos = []
        for i in range(self.num_cells):
            # Allocate PRBs in bandwidth_allocation 
            self.bandwidth_allocation[i][starting_point[i]:starting_point[i] + num_PRBs_to_allocate[i]] = 1
            # Extract initial and final pos for the assignation
            self.initial_PRB_pos.append(np.where(self.bandwidth_allocation[i] == 1)[0][0])
            self.final_PRB_pos.append(np.where(self.bandwidth_allocation[i] == 1)[0][-1])

        # Set UEs position in scenario
        self.ues_x_pos = x_pos
        self.ues_y_pos = y_pos

        # Determine Serving cell for each UE
        _ = self.assigned_UE_to_cell()

        # Call the simulator
        thr_per_cell = self.simulator()

        # Extract throughput and PRBs assigned for agent cell
        self.thr = thr_per_cell[self.agent_id]
        self.assigned_PRBs = np.sum(self.bandwidth_allocation[self.agent_id])

        # Calculate PRB position flag
        self.prb_position_flag = self.calculate_prb_position_flag()

        # Obtain the number of interference PRBs in agent cell
        self.total_interference_PRBs = self.extract_total_PRBs_interference()

        if self.total_interference_PRBs == 0:
            # No interference
            self.nbr_interference_PRBs = (self.num_cells - 1) * [0]
            self.interference_pos_flag = (self.num_cells - 1) * [0]
        else:
            # Obtain the number of interference PRBs and interference flag from each neighbour cell
            self.nbr_interference_PRBs = []
            self.interference_pos_flag = []
            for nbr_id in self.nbr_list:
                num_interf_prbs = np.sum(
                    np.multiply(self.bandwidth_allocation[self.agent_id], self.bandwidth_allocation[nbr_id]))

                if num_interf_prbs == 0:
                    # No interference with this neighbour cell
                    self.nbr_interference_PRBs.append(num_interf_prbs)
                    self.interference_pos_flag.append(0)
                else:
                    # Interference with this neighbour cell
                    self.nbr_interference_PRBs.append(num_interf_prbs)
                    self.interference_pos_flag.append(self.calculate_interference_pos_flag(nbr_id))

        # Discretization of throughput
        self.thr_discrete = self.discretize_thr(self.thr)

        # Construct Observation array: [throughput, number of assigned PRBs, PRB position flag,
        # number of interference PRBs with each neighbour cell, interference PRB flag with each neighbour cell]
        if self.num_cells == 2:
            obs = np.array([self.thr_discrete, int(self.assigned_PRBs), self.prb_position_flag,
                            int(self.nbr_interference_PRBs[0]), self.interference_pos_flag[0]
                            ])
        elif self.num_cells == 4:
            obs = np.array([self.thr_discrete, int(self.assigned_PRBs), self.prb_position_flag,
                            int(self.nbr_interference_PRBs[0]), self.interference_pos_flag[0],
                            int(self.nbr_interference_PRBs[1]), self.interference_pos_flag[1],
                            int(self.nbr_interference_PRBs[2]), self.interference_pos_flag[2]
                            ])

        self.prev_action = None

        return obs, self.total_interference_PRBs, self.thr

    def step(self, action):
        """
            Introduce the agent action in to the enviroment.
            
            Parameters:
                action (int): Agent action encoded as an integer.
            Returns:
                obs (numpy array): Observation that describes the state of the cell managed by the agent.
                Reward (int): Reward obtained by the agent's action.
                IsDone (bool): A value of True indicates that the episode has ended.
        """
        self.num_step += 1
        # Check if the maximum steps has been reached
        if self.num_step >= self.max_steps:
            IsDone = True
            # Discretize throughput
            self.thr_discrete = self.discretize_thr(self.thr)

            # Construct Observation array: [throughput, number of assigned PRBs, PRB position flag,
            # number of interference PRBs with each neighbour cell, interference PRB flag with each neighbour cell]
            if self.num_cells == 2:
                obs = np.array([self.thr_discrete, int(self.assigned_PRBs), self.prb_position_flag,
                                int(self.nbr_interference_PRBs[0]), self.interference_pos_flag[0]
                                ])
            elif self.num_cells == 4:
                obs = np.array([self.thr_discrete, int(self.assigned_PRBs), self.prb_position_flag,
                                int(self.nbr_interference_PRBs[0]), self.interference_pos_flag[0],
                                int(self.nbr_interference_PRBs[1]), self.interference_pos_flag[1],
                                int(self.nbr_interference_PRBs[2]), self.interference_pos_flag[2]
                                ])
        else:
            IsDone = False

            # The PRBs of the static cells are kept at zero, only PRBs are assigned in the agent cell
            PRBs_to_allocate = [0] * self.num_cells
            sense_to_allocate = [0] * self.num_cells

            # Decodificate the agent action
            PRBs_to_allocate[self.agent_id], sense_to_allocate[self.agent_id], shift_PRBs = self.actions_mapping.get(
                action, (0, 0, 0))

            # Allocate PRBs for the agent Cell
            allocate_prbs_succes = True
            if PRBs_to_allocate[self.agent_id] != 0:
                allocate_prbs_succes = self.PRBs_allocation(PRBs_to_allocate, sense_to_allocate)

            # Shift the assigned PRBs
            if shift_PRBs != 0:
                allocate_prbs_succes = self.PRBs_shifting(shift_PRBs)

            if allocate_prbs_succes:
                if action == 6:
                    # Not needed call the simulator
                    current_thr = self.thr
                    current_PRBs = self.assigned_PRBs
                else:
                    # Call the simulator
                    thr_per_cell = self.simulator()
                    current_thr = thr_per_cell[self.agent_id]
                    current_PRBs = np.sum(self.bandwidth_allocation[self.agent_id])

                # Obtain the number of interference PRBs in agent cell
                current_interference_PRBs = self.extract_total_PRBs_interference()

                # Obtain Reward
                self.Reward = self.calculate_reward(action, current_thr, current_PRBs, current_interference_PRBs)

                # Update Observation
                self.thr = current_thr
                self.assigned_PRBs = current_PRBs
                self.total_interference_PRBs = current_interference_PRBs

                # Calculate PRB position flag
                self.prb_position_flag = self.calculate_prb_position_flag()

                if self.total_interference_PRBs == 0:
                    # No interference
                    self.nbr_interference_PRBs = (self.num_cells - 1) * [0]
                    self.interference_pos_flag = (self.num_cells - 1) * [0]
                else:
                    # Obtain the number of interference PRBs and interference flag from each neighbour cell
                    self.nbr_interference_PRBs = []
                    self.interference_pos_flag = []
                    for nbr_id in self.nbr_list:
                        num_interf_prbs = np.sum(
                            np.multiply(self.bandwidth_allocation[self.agent_id], self.bandwidth_allocation[nbr_id]))

                        if num_interf_prbs == 0:
                            # No interference with this neighbour cell
                            self.nbr_interference_PRBs.append(num_interf_prbs)
                            self.interference_pos_flag.append(0)
                        else:
                            # Interference with this neighbour cell
                            self.nbr_interference_PRBs.append(num_interf_prbs)
                            self.interference_pos_flag.append(self.calculate_interference_pos_flag(nbr_id))

                # Discretize throughput
                self.thr_discrete = self.discretize_thr(self.thr)
                # Construct Observation array: [throughput, number of assigned PRBs, PRB position flag,
                # number of interference PRBs with each neighbour cell, interference PRB flag with each neighbour cell]
                if self.num_cells == 2:
                    obs = np.array([self.thr_discrete, int(self.assigned_PRBs), self.prb_position_flag,
                                    int(self.nbr_interference_PRBs[0]), self.interference_pos_flag[0]
                                    ])
                elif self.num_cells == 4:
                    obs = np.array([self.thr_discrete, int(self.assigned_PRBs), self.prb_position_flag,
                                    int(self.nbr_interference_PRBs[0]), self.interference_pos_flag[0],
                                    int(self.nbr_interference_PRBs[1]), self.interference_pos_flag[1],
                                    int(self.nbr_interference_PRBs[2]), self.interference_pos_flag[2]
                                    ])

            else:
                # A forbidden state has been reached
                self.Reward = -12
                # Discretize throughput
                self.thr_discrete = self.discretize_thr(self.thr)
                # Construct Observation array: [throughput, number of assigned PRBs, PRB position flag,
                # number of interference PRBs with each neighbour cell, interference PRB flag with each neighbour cell]
                if self.num_cells == 2:
                    obs = np.array([self.thr_discrete, int(self.assigned_PRBs), self.prb_position_flag,
                                    int(self.nbr_interference_PRBs[0]), self.interference_pos_flag[0]
                                    ])
                elif self.num_cells == 4:
                    obs = np.array([self.thr_discrete, int(self.assigned_PRBs), self.prb_position_flag,
                                    int(self.nbr_interference_PRBs[0]), self.interference_pos_flag[0],
                                    int(self.nbr_interference_PRBs[1]), self.interference_pos_flag[1],
                                    int(self.nbr_interference_PRBs[2]), self.interference_pos_flag[2]
                                    ])

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        if self.mode:
            # Training mode
            return obs, self.Reward, IsDone, info
        else:
            # Inference mode
            return obs, self.Reward, IsDone, info, self.total_interference_PRBs, self.thr

    def close(self):
        pass

    def discretize_thr(self, thr):
        """
        Discretizes the input throughput value (thr) into a corresponding index based on predefined intervals.
        
        Parameters:
            thr (float): Throughput value.
        Returns:
            idx_thr (int): The index of the interval in which the input throughput falls. 
        """
        # Define a fixed array of intervals that represent throughput boundaries.
        thr_intervals = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5,
                                  10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18,
                                  18.5,
                                  19, 19.5, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 35, 37, 40, 45, 50, 60, 65,
                                  70, 75, 80])

        # Find the index where the throughput value (thr) would fit into the interval array.
        # The 'side="right"' parameter ensures that if the throughput is exactly on the boundary,
        # it will be placed in the higher interval.
        idx_thr = np.searchsorted(thr_intervals, thr, side='right') - 1

        # Ensure that the index is not less than 0 to prevent accessing invalid positions.
        idx_thr = max(0, idx_thr)

        return idx_thr

    def calculate_prb_position_flag(self):
        """
        This function determines whether the allocation of PRBs is within agent_num_prbs
        positions of the left or right boundary of the array.

        Returns:
            prb_position_flag:
                0-> It is not within agent_num_prbs positions of any limit.
                1 -> PRBs are within agent_num_prbs positions of the left boundary
                2 -> PRBs are within agent_num_prbs positions of the right boundary
                3 -> PRBs are within agent_num_prbs positions of the left and right boundary.
        """
        if self.initial_PRB_pos[self.agent_id] < self.agent_num_prbs and self.final_PRB_pos[self.agent_id] <= (
                554 - self.agent_num_prbs):
            # PRBs are within agent_num_prbs positions of the left boundary.
            prb_position_flag = 1
        elif self.initial_PRB_pos[self.agent_id] >= self.agent_num_prbs and self.final_PRB_pos[self.agent_id] > (
                554 - self.agent_num_prbs):
            # PRBs are within agent_num_prbs positions of the right boundary.
            prb_position_flag = 2
        elif self.initial_PRB_pos[self.agent_id] < self.agent_num_prbs and self.final_PRB_pos[self.agent_id] > (
                554 - self.agent_num_prbs):
            # PRBs are within agent_num_prbs positions of the left and right boundary.
            prb_position_flag = 3
        else:
            # PRBs are more than agent_num_prbs positions away from either boundary.
            prb_position_flag = 0

        return prb_position_flag

    def calculate_interference_pos_flag(self, nbr_id):
        """
        This routine calculates on which side the interference is preferentially located.

        Parameters:
            nbr_id: id of the neighboring cell with which the interference is to be computed.
        Returns:
            interference_pos_flag(int):
                1 -> Most of the interfered PRBs are on the right side.
                    The best option is to move to the left or delete PRBs from right side.
                2 -> Most of the interfered PRBs are on the left side.
                    The best option is to move to the right or delete PRBs from left side.
        """

        # Case 1: The PRBs of the agent cell are completely within the interference region of the neighboring cell.
        # Calculate the difference between the positions of the PRBs on the left and right sides.
        if self.initial_PRB_pos[self.agent_id] >= self.initial_PRB_pos[nbr_id] and self.final_PRB_pos[self.agent_id] <= \
                self.final_PRB_pos[nbr_id]:
            diference_left = self.initial_PRB_pos[self.agent_id] - self.initial_PRB_pos[nbr_id]
            diference_right = self.final_PRB_pos[nbr_id] - self.final_PRB_pos[self.agent_id]

            # Case 1.1: More interference on the left side.
            if diference_right < diference_left:
                interference_pos_flag = 2  # Prefer moving to the right.

            # Case 1.2: More interference on the right side.
            elif diference_right > diference_left:
                interference_pos_flag = 1  # Prefer moving to the left.

            # Case 1.3: Interference is balanced; choose based on past interference patterns or randomly.
            else:
                if not self.interference_pos_flag:
                    # Randomly choose if no prior pattern exists.
                    interference_pos_flag = random.choice([1, 2])
                else:
                    # Choose based on historical interference flags.
                    if self.interference_pos_flag.count(1) > self.interference_pos_flag.count(2):
                        interference_pos_flag = 1  # Prefer moving to the left.
                    elif self.interference_pos_flag.count(1) < self.interference_pos_flag.count(2):
                        interference_pos_flag = 2  # Prefer moving to the right.
                    else:
                        interference_pos_flag = random.choice([1, 2])
        # Case 2: Most of the interfered PRBs are on the right side. 
        # The best option is to move to the left whenever possible.
        elif self.initial_PRB_pos[self.agent_id] < self.initial_PRB_pos[nbr_id] and self.final_PRB_pos[self.agent_id] <= \
                self.final_PRB_pos[nbr_id]:
            interference_pos_flag = 1

        # Case 3: Most of the interfered PRBs are on the left side.
        # The best option is to move to the right whenever possible.
        elif self.initial_PRB_pos[self.agent_id] >= self.initial_PRB_pos[nbr_id] and self.final_PRB_pos[self.agent_id] > \
                self.final_PRB_pos[nbr_id]:
            interference_pos_flag = 2

        # Case 4: The PRBs of the neighboring cell are within the agent cell's PRBs.
        # Calculate the difference between the left and right sides and decide the preferred direction to move.
        elif self.initial_PRB_pos[self.agent_id] < self.initial_PRB_pos[nbr_id] and self.final_PRB_pos[self.agent_id] > \
                self.final_PRB_pos[nbr_id]:
            diference_left = self.initial_PRB_pos[nbr_id] - self.initial_PRB_pos[self.agent_id]
            diference_right = self.final_PRB_pos[self.agent_id] - self.final_PRB_pos[nbr_id]

            # Case 4.1: More interference on the right side.
            if diference_left < diference_right:
                interference_pos_flag = 2  # Prefer moving to the right.

            # Case 4.2: More interference on the left side.
            elif diference_left > diference_right:
                interference_pos_flag = 1  # Prefer moving to the left.

            # Case 4.3: Balanced interference; use historical patterns or randomly decide.
            else:
                if not self.interference_pos_flag:
                    interference_pos_flag = random.choice([1, 2])
                else:
                    if self.interference_pos_flag.count(1) > self.interference_pos_flag.count(2):
                        interference_pos_flag = 1
                    elif self.interference_pos_flag.count(1) < self.interference_pos_flag.count(2):
                        interference_pos_flag = 2
                    else:
                        interference_pos_flag = random.choice([1, 2])

        return interference_pos_flag

    def calculate_reward(self, action, current_thr, current_PRBs, current_interference_PRBs):
        """
        This routine calculates the reward.
        Parameters:
            action (int): agent's action
            current_thr (float): Throughput in the current step.
            current_PRBs (int): Number of PRBs assigned in the current step.
            current_interference_PRBs: Number of interference PRBs in the current step.
        Returns:
            Reward (float).
        """
        # 1) First stage: Guarantee minimum throughput in the slice 
        if self.thr_min > current_thr:
            # The throughput is less than the minimum
            dist = (current_thr - self.thr_min) / self.thr_min
            Reward = 5 * dist - 5
        elif self.thr_min > self.thr:
            # The minimum throughput has been achieved starting from a state of throughput less than the minimum
            Reward = 3

        # 2) Second stage: Reduction of transmission power by reducing the assigned PRBs.
        elif current_PRBs < self.assigned_PRBs:
            if current_interference_PRBs < self.total_interference_PRBs:
                Reward = 3
            else:
                Reward = 2
        elif current_PRBs > self.assigned_PRBs:
            Reward = -3

        # 3) Third stage: Interference reduction
        elif current_thr > self.thr:
            Reward = 4
        elif current_thr < self.thr:
            Reward = -4
        elif current_thr == self.thr and (action == 4 or action == 5) and self.total_interference_PRBs == 0:
            Reward = -2
        elif current_thr == self.thr and (action == 4 or action == 5) and self.total_interference_PRBs != 0:
            if action == self.prev_action:
                Reward = 2
            else:
                Reward = -2
        # 4) The agent decides not to modify the bandwidth
        else:
            Reward = 1

        return Reward

    def random_starting_point(self, num_PRBs):
        """
        This routine generates an initial position for the random allocation of PRBs.
        The positions must be multiples of 10.

        Parameters:
            num_PRBs (int): Number of PRBs to assign.
        Returns:
            starting_point (int). Initial position to assignation.
        """
        num_random = np.random.randint(0, 556 - num_PRBs)
        if num_random == 0:
            return num_random
        elif num_random % 10 == 0:
            return num_random
        else:
            return num_random - (num_random % 10)

    def assigned_UE_to_cell(self):
        """
        This routine assigns each UE a server cell based on the pathloss.
        """
        # Determine serving cell for each UE
        self.dist_UEs_3D = np.zeros((self.num_cells, self.num_ues))
        self.LOS_condition_UE = np.zeros((self.num_cells, self.num_ues))
        self.L_UE = np.zeros((self.num_cells, self.num_ues))
        self.recv_power_fix = np.zeros((self.num_cells, self.num_ues))
        self.serving_cell = np.zeros(self.num_ues, dtype=int)
        for i in range(self.num_cells):
            # distance of each pixel to the different BSs
            self.dist_UEs_3D[i] = np.sqrt(((self.ues_x_pos - self.cell_x_pos[i]) * self.px_size) ** 2 +
                                          ((self.ues_y_pos - self.cell_y_pos[i]) * self.px_size) ** 2 +
                                          (self.femto_height - self.users_height) ** 2)
            # Loss in NLOS condition
            self.L_UE[i][self.LOS_condition_UE[i] == self.NLOS] = (self.femto_k_NLOS + self.femto_alpha_NLOS * np.log10(
                self.dist_UEs_3D[i][self.LOS_condition_UE[i] == self.NLOS]))
            # Loss in LOS condition
            # L_UE[i][LOS_condition_UE[i] != self.NLOS] = (self.femto_k_LOS + self.femto_alpha_LOS * np.log10(dist_UEs_3D[i][LOS_condition_UE[i] != self.NLOS]))
            # Calculate Receive Power
            self.recv_power_fix[i] = 0 - self.L_UE[i][:]
        self.serving_cell = np.argmax(self.recv_power_fix, axis=0)

        # Verify that at least all cells have a user assigned to them.
        cells_empty = np.setdiff1d(np.arange(self.num_cells), self.serving_cell)
        if len(cells_empty) > 0:
            prohibited_indices = []
            allowed_positions = np.ones(self.num_ues, dtype=bool)
            for i in cells_empty:
                if len(prohibited_indices) != 0:
                    allowed_positions[prohibited_indices] = False
                    masked_recv_power = np.where(allowed_positions, self.recv_power_fix[i][:], -np.inf)
                    UE_max_pos = np.argmax(masked_recv_power)
                    prohibited_indices.append(UE_max_pos)
                else:
                    UE_max_pos = np.argmax(self.recv_power_fix[i][:])
                    prohibited_indices.append(UE_max_pos)

                self.serving_cell[UE_max_pos] = i
        # Calculate number of UES per cell
        self.num_UEs_per_cell = np.bincount(self.serving_cell)

        return True

    def extract_total_PRBs_interference(self):
        """
        Calculate the total numer of PRBs interfered in the agent cell.
        Returns:
            total_prbs_interference (int)
        """
        # Obtain the position where the agent cell has PRBs assigned
        agent_prbs = self.bandwidth_allocation[self.agent_id, :] == 1
        # Obtain the interfering PRBs from the rest of the cells.
        delete_agent_cell = np.delete(self.bandwidth_allocation, self.agent_id, axis=0)
        other_prbs = np.any(delete_agent_cell == 1, axis=0)
        # Obtain the interference channel allocation array
        bandwidth_allocation_interference = np.zeros(self.num_prbs)
        bandwidth_allocation_interference[agent_prbs & other_prbs] = 1

        total_prbs_interference = np.sum(bandwidth_allocation_interference)

        return total_prbs_interference

    def PRBs_allocation(self, num_PRBs_to_allocate, sense_to_allocate):
        """
        This routine allocates PRBs in bandwidth_allocation  for all cells
        Parameters:
            num_PRBs_to_allocate (List):
                Contains the number of PRBs to allocate in each cell
            sense_to_allocate (int):
                Indicates on which side of the bandwidth array PRBs are added or removed.
        Returns:
            Bool:
                True if the allocation has been successful
                False if the allocation fails.
        """
        try:
            for id in self.gnb_id:
                # Add PRBs
                if num_PRBs_to_allocate[id] > 0:
                    if sense_to_allocate[id] == 1:
                        if num_PRBs_to_allocate[id] <= (self.num_prbs - 1 - self.final_PRB_pos[id]):
                            # All the PRBs can allocate on the right side of the array
                            self.bandwidth_allocation[id][
                            self.final_PRB_pos[id] + 1:self.final_PRB_pos[id] + num_PRBs_to_allocate[id] + 1] = 1
                            self.final_PRB_pos[id] = self.final_PRB_pos[id] + num_PRBs_to_allocate[id]
                        else:
                            # Number of PRBs to allocate is higher than number of PRBs available
                            return False
                    else:
                        if num_PRBs_to_allocate[id] <= self.initial_PRB_pos[id]:
                            # All the PRBs can allocate on the left side of the array
                            self.bandwidth_allocation[id][
                            self.initial_PRB_pos[id] - num_PRBs_to_allocate[id]:self.initial_PRB_pos[id]] = 1
                            self.initial_PRB_pos[id] = self.initial_PRB_pos[id] - num_PRBs_to_allocate[id]
                        else:
                            # Cant added PRBs to left side
                            return False
                # Delete PRBs
                elif num_PRBs_to_allocate[id] < 0:
                    if sense_to_allocate[id] == 1:
                        if abs(num_PRBs_to_allocate[id]) < np.sum(self.bandwidth_allocation[id]):
                            self.bandwidth_allocation[id][
                            self.final_PRB_pos[id] + 1 - abs(num_PRBs_to_allocate[id]):self.final_PRB_pos[id] + 1] = 0
                            self.final_PRB_pos[id] = self.final_PRB_pos[id] - abs(num_PRBs_to_allocate[id])
                        else:
                            # Number of PRBs to delete is higher than number of PRBs assignated
                            return False
                    else:
                        if abs(num_PRBs_to_allocate[id]) < np.sum(self.bandwidth_allocation[id]):
                            self.bandwidth_allocation[id][
                            self.initial_PRB_pos[id]:self.initial_PRB_pos[id] + abs(num_PRBs_to_allocate[id])] = 0
                            self.initial_PRB_pos[id] = self.initial_PRB_pos[id] + abs(num_PRBs_to_allocate[id])
                        else:
                            # Number of PRBs to delete is higher than number of PRBs assignated
                            return False
            return True
        except Exception as e:
            print("PRBS_allocation(): ERROR")
            return False

    def PRBs_shifting(self, shift_PRBs):
        """
        This routine shifts the assigned PRBs in agent cell.
        Parameters:
            shift_PRBs (int)
                1: Shift to right
                -1: Shift to left
        Returns:
            bool:
                True: Shifting has been successful
                False: Shifting fails
        """

        if shift_PRBs > 0:  # Shift right
            # Check if shifting right would exceed the maximum PRB position limit (554).
            if self.final_PRB_pos[self.agent_id] + shift_PRBs > 554:
                return False
            else:
                self.bandwidth_allocation[self.agent_id] = np.roll(self.bandwidth_allocation[self.agent_id], shift_PRBs)
                self.initial_PRB_pos[self.agent_id] += shift_PRBs
                self.final_PRB_pos[self.agent_id] += shift_PRBs
        else:  # Shift left
            # Check if shifting left would go below the minimum PRB position limit (0).
            if self.initial_PRB_pos[self.agent_id] - abs(shift_PRBs) < 0:
                return False
            else:
                self.bandwidth_allocation[self.agent_id] = np.roll(self.bandwidth_allocation[self.agent_id],
                                                                   - abs(shift_PRBs))
                self.initial_PRB_pos[self.agent_id] -= abs(shift_PRBs)
                self.final_PRB_pos[self.agent_id] -= abs(shift_PRBs)
        return True

    def simulator(self):
        """
        This routine implements a 5G NG-RAN simulator that calculates the throughput of all UES 
        and returns the UE throughput at the edge of each cell in the scenario.
        
        Returns:
            thr_UE_edge (array).
        """
        # Initialize variables
        nosaturated_load_cell = np.zeros((self.num_cells, self.num_prbs))
        interf = np.zeros((self.num_prbs, self.num_ues))
        SINR_PRB = np.zeros((self.num_prbs, self.num_ues))
        sp_eff_prb = np.zeros((self.num_prbs, self.num_ues))
        sp_eff_prb_modified = np.zeros((self.num_prbs, self.num_ues))
        throughput_prb = np.zeros((self.num_prbs, self.num_ues))
        SINR = np.zeros(self.num_ues)
        sp_eff_UEs = np.zeros(self.num_ues)
        throughput_UEs = np.zeros(self.num_ues)
        cell_throughput = np.zeros(self.num_cells)
        thr_UE_edge = np.full(self.num_cells, np.inf)

        # 1) Calculate the received power at each UE of each cell.
        cell_power = 10 * np.log10(self.power_per_prb * np.sum(self.bandwidth_allocation, axis=1))
        recv_power_UE = cell_power[:, np.newaxis] - self.L_UE

        # 2) Determine possible serving cell for each UE (association based on received power)
        for j in range(self.num_ues):
            # Calculate cell load (no saturated, i.e. it can be above one)
            nosaturated_load_cell[self.serving_cell[j], :] = nosaturated_load_cell[self.serving_cell[j], :] + (
                    (self.br_embb / self.femto_avg_SE) / (np.sum(
                self.bandwidth_allocation[self.serving_cell[j], :]) * self.bw_prb)) * self.bandwidth_allocation[
                                                                                      self.serving_cell[j], :]

        # Calculate cell load
        load_cell = np.minimum(nosaturated_load_cell, 1)

        # 3) Calculate SINR, SE in each UE with definitive serving cell
        for j in range(self.num_ues):
            PRBs_idx = np.where(self.bandwidth_allocation[self.serving_cell[j], :])[0]
            for k in PRBs_idx:
                interf[k, j] = np.sum(
                    np.power(10, 0.1 * recv_power_UE[:, j]) * self.bandwidth_allocation[:, k] * load_cell[:,
                                                                                                k]) - np.power(10, 0.1 *
                                                                                                               recv_power_UE[
                                                                                                                   self.serving_cell[
                                                                                                                       j], j]) * \
                               load_cell[self.serving_cell[j], k]
                # Calculate SINR per PRB
                SINR_PRB[k][j] = np.power(10, 0.1 * recv_power_UE[self.serving_cell[j], j]) / (
                        interf[k, j] + np.power(10, 0.1 * self.ue_pnoise_ch_femto))
                # Calculate SE per PRB
                if SINR_PRB[k][j] <= self.SINRmin:
                    sp_eff_prb[k][j] = 0
                elif SINR_PRB[k][j] < self.SINRmax:
                    sp_eff_prb[k][j] = self.alpha_thr * math.log2(1 + SINR_PRB[k][j])
                else:
                    sp_eff_prb[k][j] = self.Smax
                sp_eff_prb_modified[k][j] = max(sp_eff_prb[k][j], self.scheduling_SE_param)

                # Calculate throughput per PRB               
                throughput_prb[k][j] = sp_eff_prb[k][j] * self.bw_prb / self.num_UEs_per_cell[self.serving_cell[j]]

            # Calculate wideband SINR
            SINR[j] = np.sum(SINR_PRB[:, j]) / np.sum(self.bandwidth_allocation[self.serving_cell[j]])
            # Calculate average SE
            sp_eff_UEs[j] = np.sum(sp_eff_prb[:][j]) / np.sum(self.bandwidth_allocation[self.serving_cell[j]])
            # Calculate total throughput
            throughput_UEs[j] = np.sum(throughput_prb[:, j])

            cell_throughput[self.serving_cell[j]] = cell_throughput[self.serving_cell[j]] + throughput_UEs[j]

            # Get the throughput of the worst user (UE at the edge) for each cell.
            if j == 0:
                thr_UE_edge[self.serving_cell[j]] = throughput_UEs[j]
            else:
                thr_UE_edge[self.serving_cell[j]] = np.minimum(thr_UE_edge[self.serving_cell[j]], throughput_UEs[j])

        return thr_UE_edge
