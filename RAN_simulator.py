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
import math
import random


class RAN_Simulator:
    """
    This class implements a Radio Access Network Simulator capable
    of modifying the bandwidth of cells according to the actions entered
    by an agent. The class returns the observations necessary for an agent
    to make decisions in each cell.
    """

    def __init__(self, num_cells_scenario):
        """
        Initialize the Radio Access Network Simulator.

        Parameters:
        num_cells_scenario (int): Identifier for the scenario configuration (2 or 4).
        """
        if num_cells_scenario == 2:
            scenario_config_file = 'config_files/scenario_config_2_cells.csv'
            simulator_config_file = 'config_files/simulator_config_2_cells.csv'
        elif num_cells_scenario == 4:
            scenario_config_file = 'config_files/scenario_config_4_cells.csv'
            simulator_config_file = 'config_files/simulator_config_4_cells.csv'

        # Load scenario configuration from CSV
        scenario_config_df = pd.read_csv(scenario_config_file, sep=';')

        # Scenario configuration variables
        self.agent_num_prbs = scenario_config_df['agent_num_prbs'].iloc[0]
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

    def start(self, num_PRBs_to_allocate, starting_point, ues_pos):
        """
        This routine allocates PRBs at the specified locations and assigns 
        UEs to serving cells. Returns observations that describe the state of each cell.

        Parameters:
            num_PRBs_to_allocate (list): List with the PRBs to be assigned to each cell
            starting_point (list): List with the positions where the allocation of PRBs begins in each cell
            ues_pos (numpy array): Numpy array with the positions of the UEs in the x and y dimension
        Returns:
            observations (numpy array): Array containing the observations from all cells.
            initial_PRB_pos (numpy array): Array with the positions where the assignment begins in each cell.
            final_PRB_pos (numpy array): Array with the positions where the assignment ends in each cell.
            num_total_prbs_interference (numpy array): Array with the number of PRBs with interference in each cell
            thr (numpy array): Array with the throughput values ​​(float) of each cell
            serving_cell (list): List with the serving cell of each UE
            thr_UE_ID (list):  List with the ID of the UEs with the worst throughput in each cell.
        """
        # Initialization bandwidth_allocation  as zeros numpy arrays
        self.bandwidth_allocation = np.zeros((self.num_cells, self.num_prbs))
        self.initial_PRB_pos = []
        self.final_PRB_pos = []
        # Assignment of PRBs in the initial positions
        for i in range(self.num_cells):
            self.bandwidth_allocation[i][starting_point[i]:starting_point[i] + num_PRBs_to_allocate[i]] = 1
            # Extract initial and final pos for the assignation
            self.initial_PRB_pos.append(np.where(self.bandwidth_allocation[i] == 1)[0][0])
            self.final_PRB_pos.append(np.where(self.bandwidth_allocation[i] == 1)[0][-1])

        # Set UEs position in scenario
        self.ues_x_pos = ues_pos[0][:]
        self.ues_y_pos = ues_pos[1][:]

        # Determine Serving cell for each UE
        _ = self.assigned_UE_to_cell()

        # Call the simulator
        self.thr, self.thr_UE_ID = self.run_simulator()
        self.thr_discrete = self.discretize_thr(self.thr)
        self.PRBs_assigned = list(np.sum(self.bandwidth_allocation, axis=1))

        # Calculate interference for each cell and construct observation array
        if self.num_cells == 2:
            observations = np.zeros((self.num_cells, 5), dtype=int)
        else:
            observations = np.zeros((self.num_cells, 9), dtype=int)

        self.num_total_prbs_interference = []
        for agent_id in range(self.num_cells):
            self.num_total_prbs_interference.append(self.extract_total_PRBs_interference(agent_id))
            interference_PRBs = []
            interference_pos_flag = []
            nbr_list = [nbr_id for nbr_id in self.gnb_id if nbr_id != agent_id]
            for nbr_id in nbr_list:
                # Obtain the number of interference PRBs in agent cell
                interf_prbs = np.sum(
                    np.multiply(self.bandwidth_allocation[agent_id], self.bandwidth_allocation[nbr_id]))

                # Obtain the number of interference PRBs and interference flag
                if interf_prbs == 0:
                    interference_PRBs.append(interf_prbs)
                    interference_pos_flag.append(0)
                else:
                    interference_PRBs.append(interf_prbs)
                    interference_pos_flag.append(self.calculate_interference_pos_flag(agent_id, nbr_id))

            # Calculate PRB position flag
            prb_position_flag = self.calculate_prb_position_flag(agent_id)

            if self.num_cells == 2:
                observations[agent_id] = self.thr_discrete[agent_id], int(
                    self.PRBs_assigned[agent_id]), prb_position_flag, int(interference_PRBs[0]), interference_pos_flag[
                                             0]
            else:
                observations[agent_id] = self.thr_discrete[agent_id], self.PRBs_assigned[
                    agent_id], prb_position_flag, int(interference_PRBs[0]), interference_pos_flag[0], int(
                    interference_PRBs[1]), interference_pos_flag[1], int(interference_PRBs[2]), interference_pos_flag[2]

        return observations, np.array(self.initial_PRB_pos), np.array(self.final_PRB_pos), np.array(
            self.num_total_prbs_interference), np.array(self.thr), self.serving_cell, self.thr_UE_ID

    def run(self, actions):
        """
        This routine introduces the agent's actions into 
        each cell of the scenario and modifies the cells' bandwidth according to these actions.

        Parameters:
            actions (list): List with the agent's actions
        Returns:
            observations (numpy array): Array containing the observations from all cells.
            initial_PRB_pos (numpy array): Array with the positions where the assignment begins in each cell.
            final_PRB_pos (numpy array): Array with the positions where the assignment ends in each cell.
            num_total_prbs_interference (numpy array): Array with the number of PRBs with interference in each cell
            thr (numpy array): Array with the throughput values (float) of each cell
        """
        PRBs_to_allocate = []
        sense_to_allocate = []
        shift_PRBs = []
        for i in actions:
            prbs, sense, shift = self.actions_mapping.get(i, (0, 0, 0))
            PRBs_to_allocate.append(prbs)
            sense_to_allocate.append(sense)
            shift_PRBs.append(shift)

        # Allocate PRBs
        allocate_prbs_succes = self.PRBs_allocation(PRBs_to_allocate, sense_to_allocate)

        # Shift PRBs
        shift_prbs_succes = self.PRBs_shifting(shift_PRBs)

        if allocate_prbs_succes and shift_prbs_succes:
            self.thr, _ = self.run_simulator()
            self.thr_discrete = self.discretize_thr(self.thr)
            self.PRBs_assigned = list(np.sum(self.bandwidth_allocation, axis=1))

        # Calculate interference for each cell and construct observation array
        if self.num_cells == 2:
            observations = np.zeros((self.num_cells, 5), dtype=int)
        else:
            observations = np.zeros((self.num_cells, 9), dtype=int)

        self.num_total_prbs_interference = []
        for agent_id in range(self.num_cells):
            self.num_total_prbs_interference.append(self.extract_total_PRBs_interference(agent_id))
            interference_PRBs = []
            interference_pos_flag = []
            nbr_list = [nbr_id for nbr_id in self.gnb_id if nbr_id != agent_id]
            for nbr_id in nbr_list:
                # Obtain the number of interference PRBs in agent cell
                interf_prbs = np.sum(
                    np.multiply(self.bandwidth_allocation[agent_id], self.bandwidth_allocation[nbr_id]))

                # Obtain the number of interference PRBs and interference flag
                if interf_prbs == 0:
                    interference_PRBs.append(interf_prbs)
                    interference_pos_flag.append(0)
                else:
                    interference_PRBs.append(interf_prbs)
                    interference_pos_flag.append(self.calculate_interference_pos_flag(agent_id, nbr_id))

            # Calculate PRB position flag
            prb_position_flag = self.calculate_prb_position_flag(agent_id)

            if self.num_cells == 2:
                observations[agent_id] = self.thr_discrete[agent_id], int(
                    self.PRBs_assigned[agent_id]), prb_position_flag, int(interference_PRBs[0]), interference_pos_flag[
                                             0]
            else:
                observations[agent_id] = self.thr_discrete[agent_id], self.PRBs_assigned[
                    agent_id], prb_position_flag, int(interference_PRBs[0]), interference_pos_flag[0], int(
                    interference_PRBs[1]), interference_pos_flag[1], int(interference_PRBs[2]), interference_pos_flag[2]

        return observations, np.array(self.initial_PRB_pos), np.array(self.final_PRB_pos), np.array(
            self.num_total_prbs_interference), np.array(self.thr)

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
        idx_thr = np.maximum(0, idx_thr)

        return idx_thr

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
        for i in range(self.num_cells):

            if shift_PRBs[i] > 0:  # Shift right
                if self.final_PRB_pos[i] + shift_PRBs[i] > 554:
                    return False
                else:
                    self.bandwidth_allocation[i] = np.roll(self.bandwidth_allocation[i], shift_PRBs[i])
                    self.initial_PRB_pos[i] += shift_PRBs[i]
                    self.final_PRB_pos[i] += shift_PRBs[i]
            elif shift_PRBs[i] < 0:
                if self.initial_PRB_pos[i] - abs(shift_PRBs[i]) < 0:
                    return False
                else:
                    self.bandwidth_allocation[i] = np.roll(self.bandwidth_allocation[i], - abs(shift_PRBs[i]))
                    self.initial_PRB_pos[i] -= abs(shift_PRBs[i])
                    self.final_PRB_pos[i] -= abs(shift_PRBs[i])
        return True

    def assigned_UE_to_cell(self):
        """
            This routine assigns each UE a server cell based on the pathloss.
        """
        # Determine serving Bs for each UE
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
            for i in cells_empty:
                UE_max_pos = np.argmax(self.recv_power_fix[i][:])
                self.serving_cell[UE_max_pos] = i
        # Calculate number of UES per BS
        self.num_UEs_per_cell = np.bincount(self.serving_cell)

        return True

    def extract_total_PRBs_interference(self, cell_id):
        """
        Calculate the total numer of PRBs interfered in a cell.
        
        Parameters:
            cell_id: cell ID to calculate PRBs interfered.
        Returns:
            total_prbs_interference (int)
        """
        # Obtain the position where the agent cell has PRBs assigned
        agent_prbs = self.bandwidth_allocation[cell_id, :] == 1
        # Obtain the interfering PRBs from the rest of the cells.
        delete_agent_cell = np.delete(self.bandwidth_allocation, cell_id, axis=0)
        other_prbs = np.any(delete_agent_cell == 1, axis=0)
        # Obtain the interference channel allocation array
        bandwidth_allocation_interference = np.zeros(self.num_prbs)
        bandwidth_allocation_interference[agent_prbs & other_prbs] = 1

        total_prbs_interference = np.sum(bandwidth_allocation_interference)

        return total_prbs_interference

    def calculate_prb_position_flag(self, cell_id):
        """
        This function determines whether the allocation of PRBs is within agent_num_prbs positions of the left
        or right boundary of the array.

        Outputs:
            prb_position_flag:
                0-> It is not within agent_num_prbs positions of any limit.
                1 -> PRBs are within agent_num_prbs positions of the left boundary
                2 -> PRBs are within agent_num_prbs positions of the right boundary
                3 -> PRBs are within agent_num_prbs positions of the left and right boundary.
        """
        if self.initial_PRB_pos[cell_id] < self.agent_num_prbs and self.final_PRB_pos[cell_id] <= (
                554 - self.agent_num_prbs):
            # PRBs are within agent_num_prbs positions of the left boundary.
            prb_position_flag = 1
        elif self.initial_PRB_pos[cell_id] >= self.agent_num_prbs and self.final_PRB_pos[cell_id] > (
                554 - self.agent_num_prbs):
            # PRBs are within agent_num_prbs positions of the right boundary.
            prb_position_flag = 2
        elif self.initial_PRB_pos[cell_id] < self.agent_num_prbs and self.final_PRB_pos[cell_id] > (
                554 - self.agent_num_prbs):
            # PRBs are within agent_num_prbs positions of the left and right boundary.
            prb_position_flag = 3
        else:
            # PRBs are more than agent_num_prbs positions away from either boundary.
            prb_position_flag = 0

        return prb_position_flag

    def calculate_interference_pos_flag(self, agent_id, nbr_id):
        """
        This routine calculates on which side the interference is preferentially located.

        Parameters:
            agent_id: ID of the agent cell.
            nbr_id: ID of the neighboring cell with which the interference is to be computed.
        Returns:
            interference_pos_flag(int):
                1 -> Most of the interfered PRBs are on the right side.
                    The best option is to move to the left or delete PRBs from right side.
                2 -> Most of the interfered PRBs are on the left side.
                    The best option is to move to the right or delete PRBs from left side.
        """

        # Case 1: The PRBs of the agent cell are completely within the interference region of the neighboring cell.
        # Calculate the difference between the positions of the PRBs on the left and right sides.
        if self.initial_PRB_pos[agent_id] >= self.initial_PRB_pos[nbr_id] and self.final_PRB_pos[agent_id] <= \
                self.final_PRB_pos[nbr_id]:
            diference_left = self.initial_PRB_pos[agent_id] - self.initial_PRB_pos[nbr_id]
            diference_right = self.final_PRB_pos[nbr_id] - self.final_PRB_pos[agent_id]

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
        elif self.initial_PRB_pos[agent_id] < self.initial_PRB_pos[nbr_id] and self.final_PRB_pos[agent_id] <= \
                self.final_PRB_pos[nbr_id]:
            interference_pos_flag = 1

        # Case 3: Most of the interfered PRBs are on the left side.
        # The best option is to move to the right whenever possible.
        elif self.initial_PRB_pos[agent_id] >= self.initial_PRB_pos[nbr_id] and self.final_PRB_pos[agent_id] > \
                self.final_PRB_pos[nbr_id]:
            interference_pos_flag = 2

        # Case 4: The PRBs of the neighboring cell are within the agent cell's PRBs.
        # Calculate the difference between the left and right sides and decide the preferred direction to move.
        elif self.initial_PRB_pos[agent_id] < self.initial_PRB_pos[nbr_id] and self.final_PRB_pos[agent_id] > \
                self.final_PRB_pos[nbr_id]:
            diference_left = self.initial_PRB_pos[nbr_id] - self.initial_PRB_pos[agent_id]
            diference_right = self.final_PRB_pos[agent_id] - self.final_PRB_pos[nbr_id]

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

    def run_simulator(self):
        """
        This routine implements a 5G NG-RAN simulator that calculates the throughput of all UES 
        and returns the UE throughput at the edge of each cell in the scenario.
        
        Returns:
            thr_UE_edge (array): Array with throughput values.
            thr_UE_ID (array): Array with the UE IDs with the worst throughput in each cell.
        """

        # Initialize variables
        nosaturated_load_cell = np.zeros((self.num_cells, self.num_prbs))
        interf = np.zeros((self.num_prbs, self.num_ues))
        SINR_PRB = np.zeros((self.num_prbs, self.num_ues))
        sp_eff_prb = np.zeros((self.num_prbs, self.num_ues))
        sp_eff_prb_modified = np.zeros((self.num_prbs, self.num_ues))
        throughput_prb = np.zeros((self.num_prbs, self.num_ues))
        prb_allocated_UE = np.zeros((self.num_cells, self.num_ues))
        SINR = np.zeros(self.num_ues)
        sp_eff_UEs = np.zeros(self.num_ues)
        throughput_UEs = np.zeros(self.num_ues)
        cell_throughput = np.zeros(self.num_cells)
        thr_UE_edge = np.full(self.num_cells, np.inf)
        thr_UE_ID = np.full(self.num_cells, np.inf)

        # 1) Calculate the received power at each user of each base station for interference
        cell_power = 10 * np.log10(self.power_per_prb * np.sum(self.bandwidth_allocation, axis=1))
        recv_power_UE = cell_power[:, np.newaxis] - self.L_UE

        # 2) Determine possible serving BS for each UE (association based on received power)
        for j in range(self.num_ues):
            # Calculate BS load (no saturated, i.e. it can be above one)
            nosaturated_load_cell[self.serving_cell[j], :] = nosaturated_load_cell[self.serving_cell[j], :] + (
                        (self.br_embb / self.femto_avg_SE) / (np.sum(
                    self.bandwidth_allocation[self.serving_cell[j], :]) * self.bw_prb)) * self.bandwidth_allocation[
                                                                                          self.serving_cell[j], :]
        # Calculate BS load
        load_cell = np.minimum(nosaturated_load_cell, 1)

        # 3) Calculate SINR, SE in each UE with definitive serving BS
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
                # Calculate SINR per channel
                SINR_PRB[k][j] = np.power(10, 0.1 * recv_power_UE[self.serving_cell[j], j]) / (
                            interf[k, j] + np.power(10, 0.1 * self.ue_pnoise_ch_femto))
                # calculate SE per channel
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
                thr_UE_ID[self.serving_cell[j]] = j
            else:
                if throughput_UEs[j] < thr_UE_edge[self.serving_cell[j]]:
                    thr_UE_edge[self.serving_cell[j]] = throughput_UEs[j]
                    thr_UE_ID[self.serving_cell[j]] = j

        return thr_UE_edge, thr_UE_ID
