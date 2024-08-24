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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def plot_prbs_array(num_cells, starting_point, end_point, title, num_prbs_to_allocate):
    
    # Create a PRB (Physical Resource Block) allocation array
    bandwidth_allocation = np.zeros((num_cells, 555)).astype(int)
    for i in range(num_cells):
        # Mark the allocated PRBs in the array for each cell
        bandwidth_allocation[i, int(starting_point[i]):int(end_point[i])] = 1

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(num_cells, 1, figsize=(8, num_cells * 2), sharex=True)

    # Define colors for the values 0 (unallocated) and 1 (allocated)
    colors_cells = ['blue', 'orange', 'green', 'red']
    
    # Check if the number of cells exceeds the available colors
    if num_cells > len(colors_cells):
        raise ValueError("The number of cells exceeds the number of available colors")

    # Iterate over each row of the PRB allocation array
    for i in range(bandwidth_allocation.shape[0]):
        color = colors_cells[i % len(colors_cells)]
        # Create an array of colors based on the values in the allocation array
        color_array = ['white' if val == 0 else color for val in bandwidth_allocation[i]]
        # Draw a rectangle for each PRB based on its allocation status
        for j, color in enumerate(color_array):
            axs[i].add_patch(plt.Rectangle((j, 0), 1, 1, color=color))

        # Configure axes and labels
        axs[i].set_xlim(0, len(bandwidth_allocation[i]))
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axs[i].set_ylim(0, 1)
        axs[i].set_yticks([])
        axs[i].set_title(f"Cell SC_{i} ({num_prbs_to_allocate[i]} PRBs)", fontsize=12)

        # Add a grid to the subplot
        axs[i].grid(True, color='black', linewidth=0.5, linestyle='-')

    # Adjust the layout of the plots and display the figure
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_thr(thr, actions):

    # Define the colors corresponding to each action
    colores = ['red', 'yellow', 'green', 'blue', 'purple', 'orange', 'white']
    # Define the names of the actions to be displayed in the legend
    name_action = ['Delete PRBs from left', 'Delete PRBs from right', 'Add PRBs from left',
                   'Add PRBs from right', 'Shifts PRBs to right', 'Shifts PRBs to left', 'Do nothing']
    # Determine the layout of subplots based on the number of rows in thr
    if thr.shape[0] == 2:
        fig, axs = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    elif thr.shape[0] == 4:
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    else:
        # Handle other sizes (optional)
        fig, axs = plt.subplots(thr.shape[0], 1, figsize=(10, 6), sharex=True)

    # Ensure axs is an array for easy iteration
    axs = np.array(axs).reshape(-1)
    # Iterate over each throughput array in thr
    for i in range(thr.shape[0]):
        # Plot the throughput line graph for each subplot
        axs[i].plot(thr[i], color='black')
        # Draw background rectangles corresponding to the actions taken at each step
        for j, action in enumerate(actions[i]):
            if j != 0:
                axs[i].axvspan(j - 1, j, color=colores[int(action)], alpha=0.3)

        # Draw a horizontal line at y = 5 to indicate a threshold
        axs[i].axhline(y=5, color='r', linestyle='--')
        # Annotate the first and last throughput values on the graph
        axs[i].text(0, thr[i][0], f'{thr[i][0]:.2f}', verticalalignment='bottom', color='black', fontsize=10)
        axs[i].text(len(thr[i]) - 2, thr[i][-1], f'{thr[i][-1]:.2f}', verticalalignment='bottom', color='black',
                    fontsize=10)

        # Additional plot adjustments
        axs[i].set_xlabel('Step')
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axs[i].set_ylabel('Throughput (Mbps)')
        axs[i].set_title('Cell SC_{}'.format(i, thr[i][0], thr[i][-1]), fontsize=10)
        axs[i].set_xlim(0, len(thr[i]) - 1)
    
    # Create a list of patches for the legend, one for each action color
    patches = [mpatches.Patch(color=colores[i], label=name_action[i]) for i in range(len(colores))]

    # Add the legend to the main figure
    fig.legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

    # Adjust the layout of the subplots and add space for the legend
    plt.suptitle('Evolution of throughput')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def plot_assigned_prbs(prbs, actions):

    # Define the colors corresponding to each action
    colores = ['red', 'yellow', 'green', 'blue', 'purple', 'orange', 'white']
    name_action = ['Delete PRBs from left', 'Delete PRBs from right', 'Add PRBs from left',
                   'Add PRBs from right', 'Shifts PRBs to right', 'Shifts PRBs to left', 'Do nothing']
    
    if prbs.shape[0] == 2:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    elif prbs.shape[0] == 4:
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    else:
        fig, axs = plt.subplots(prbs.shape[0], 1, figsize=(10, 6), sharex=True)

    axs = np.array(axs).reshape(-1)

    for i in range(prbs.shape[0]):
        # Plot the PRB data as a black line
        axs[i].plot(prbs[i], color='black')

        # Draw rectangles in the background of the plot according to the actions
        for j, action in enumerate(actions[i]):
            if j != 0:
                # Highlight the region between steps based on the action taken
                axs[i].axvspan(j - 1, j, color=colores[int(action)], alpha=0.3)
        # Add text labels to the start and end of the PRB series
        axs[i].text(0, prbs[i][0], f'{int(prbs[i][0])}', verticalalignment='bottom', color='black', fontsize=10)
        axs[i].text(len(prbs[i]) - 2, prbs[i][-1], f'{int(prbs[i][-1])}', verticalalignment='bottom', color='black',
                    fontsize=10)

        axs[i].set_xlabel('Step')
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axs[i].set_ylabel('Num PRBs assigned')
        axs[i].set_title('Cell SC_{}'.format(i, int(prbs[i][0]), int(prbs[i][-1])), fontsize=14)
        
        # Set the x-axis limits
        axs[i].set_xlim(0, len(prbs[i]) - 1)
    
    # Create patches for the legend corresponding to each action
    patches = [mpatches.Patch(color=colores[i], label=name_action[i]) for i in range(len(colores))]

    # Add the legend to the figure 
    fig.legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

    # Adjust the layout of the subplots and leave space for the legend
    plt.suptitle('Evolution of allocated PRBs')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def plot_interference_prbs(prbs, actions):
    
    # Define the colors corresponding to each action
    colores = ['red', 'yellow', 'green', 'blue', 'purple', 'orange', 'white']
    name_action = ['Delete PRBs from left', 'Delete PRBs from right', 'Add PRBs from left',
                   'Add PRBs from right', 'Shifts PRBs to right', 'Shifts PRBs to left', 'Do nothing']
    if prbs.shape[0] == 2:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    elif prbs.shape[0] == 4:
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    else:
        fig, axs = plt.subplots(prbs.shape[0], 1, figsize=(10, 6), sharex=True)

    axs = np.array(axs).reshape(-1)
    for i in range(prbs.shape[0]):
        axs[i].plot(prbs[i], color='black')

        # Draw rectangles in the background of the plot according to the actions
        for j, action in enumerate(actions[i]):
            if j != 0:
                axs[i].axvspan(j - 1, j, color=colores[int(action)], alpha=0.3)

        axs[i].text(0, prbs[i][0], f'{int(prbs[i][0])}', verticalalignment='bottom', color='black', fontsize=10)
        axs[i].text(len(prbs[i]) - 2, prbs[i][-1], f'{int(prbs[i][-1])}', verticalalignment='bottom', color='black',
                    fontsize=10)

        axs[i].set_xlabel('Step')
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axs[i].set_ylabel('Num of interference')
        axs[i].set_title('Cell SC_{}'.format(i, int(prbs[i][0]), int(prbs[i][-1])), fontsize=14)
        axs[i].set_xlim(0, len(prbs[i]) - 1)
    patches = [mpatches.Patch(color=colores[i], label=name_action[i]) for i in range(len(colores))]

    fig.legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

    plt.suptitle('Evolution of PRBs with interference')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def plot_scenario(num_cells, num_ues, ues_x_pos, ues_y_pos, serving_BS, thr, thr_ID):
    
    if num_cells == 2:
        title='Scenario A'
        cell_x_pos = [15, 85]
        cell_y_pos = [15, 85]
    else:
        title='Scenario B'
        cell_x_pos = [15, 15, 85, 85]
        cell_y_pos = [15, 85, 85, 15]
    
    scenario_x_size = 100
    scenario_y_size = 100
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    # Load custom base station icons
    base_station_icon_1 = plt.imread('utils/icon_sc_0.png')
    base_station_icon_2 = plt.imread('utils/icon_sc_1.png')
    base_station_icon_3 = plt.imread('utils/icon_sc_2.png')
    base_station_icon_4 = plt.imread('utils/icon_sc_3.png')

    # Plot base stations with custom icons and labels
    for i in range(num_cells):
        if i == 0:
            imagebox = OffsetImage(base_station_icon_1, zoom=0.1)
        elif i == 1:
            imagebox = OffsetImage(base_station_icon_2, zoom=0.15)
        elif i == 2:
            imagebox = OffsetImage(base_station_icon_3, zoom=0.15)
        elif i == 3:
            imagebox = OffsetImage(base_station_icon_4, zoom=0.15)

        ab = AnnotationBbox(imagebox, (cell_x_pos[i], cell_y_pos[i]), frameon=False)
        ax.add_artist(ab)
        ax.text(cell_x_pos[i] + 3, cell_y_pos[i], f'SC_{i}', fontsize=10, verticalalignment='center')

    # Plot UEs with colors corresponding to serving base station and user ID as text
    for ue_idx in range(num_ues):
        bs_idx = serving_BS[ue_idx]
        ax.scatter(ues_x_pos[ue_idx], ues_y_pos[ue_idx], color='C{}'.format(bs_idx), marker='o')

    # Plot thr of users at the edge

    for i in range(len(thr_ID)):
        bs_idx = serving_BS[int(thr_ID[i])]
        ax.text(ues_x_pos[int(thr_ID[i])] + 0.4, ues_y_pos[int(thr_ID[i])] + 0.4, f'{thr[i]:.2f}Mbps', fontsize=10,
                ha='right', color='C{}'.format(bs_idx))

    # Create legend based on unique base station colors
    handles = []
    labels = []
    for i in range(num_cells):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='C{}'.format(i), markersize=10))
    # labels.append('SC_{}'.format(i))

    # ax.legend(handles, labels, loc='upper right')

    ax.set_xlim(0, scenario_x_size)
    ax.set_ylim(0, scenario_y_size)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('{}'.format(title))
    ax.grid(True)

    plt.show()

