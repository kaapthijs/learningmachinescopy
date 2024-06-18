import pickle
import numpy as np
import cv2
import os
import itertools
import random

from data_files import FIGRURES_DIR
from data_files import RESULT_DIR

from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

# GLOBAL VARIABLES
# Define number of sensors used
NUM_SENSORS = 3

# Define the possible actions
ACTIONS = ['left', 'forward', 'right']
NUM_ACTIONS = len(ACTIONS)

# Define number of InfraRed bins where sensor falls in
IR_BINS = 6    # sensor value could be 0,1,2,3
BIN_THRESHOLDS = [4,7,10,15,25]
BIN_THRESHOLDS_HARDWARE = [-1,15,100]

# Define number of Greenness bins
GREEN_BINS = 3 # 0,1,2
GREEN_THRESHOLDS = [5,10,15]
GREEN_THRESHOLDS_HARDWARE = [5,10,15]

# Define rewards of moving
FORWARD_REWARD = 11
HIT_PENALTY = -50
COLLISION_STATE = IR_BINS-1

# Define global constants for movement settings
FORWARD_SPEED_LEFT = 50
FORWARD_SPEED_RIGHT = 50
FORWARD_DURATION = 300

RIGHT_SPEED_LEFT = 60
RIGHT_SPEED_RIGHT = -30
RIGHT_DURATION = 300

LEFT_SPEED_LEFT = -30
LEFT_SPEED_RIGHT = 60
LEFT_DURATION = 300

# Functions for loading and saving q-table
def load_q_table(q_table_path):
    with open(str(RESULT_DIR / q_table_path), 'rb') as f:
        return pickle.load(f)
    
def save_q_table(q_table, q_table_path):
    with open(str(RESULT_DIR / q_table_path), 'wb') as f:
        pickle.dump(q_table, f)


# Initialize Q-table
def initialize_q_table(q_table_path, num_bins=IR_BINS, num_sensors=NUM_SENSORS, num_actions=NUM_ACTIONS):
    """Load the Q-table from a file if it exists; otherwise, initialize a new Q-table."""
    if os.path.exists(q_table_path):
        with open(q_table_path, 'rb') as f:
            q_table = pickle.load(f)
    else:
        q_table = {}
        for state in itertools.product(range(num_bins), repeat=num_sensors):
            q_table[state] = [0.0 for _ in range(num_actions)]
            
        save_q_table(q_table, q_table_path)  # Save the new Q-table for the first time
    
    return q_table

# Visualize Q-table
def print_q_table(q_table, num_entries=10):
    print(f"{'State':<15} | {'Q-values (forward, left, right)':<30}")
    print("-" * 50)
    for i, (state, q_values) in enumerate(q_table.items()):
        print(f"{str(state):<15} | {q_values}")
        if i >= num_entries - 1:
            break

# Function that writes values into csv in directory
def write_csv(data_values, dir_str):
    with open(dir_str,'a') as f:
                for item in data_values:
                    f.write(str(item))
                    f.write(';')
                f.write('\n')

# Function that retrieves action from q_table
# if epsilon is given, retrieve with creativity, else retrive argmax
def get_action(q_table, state, epsilon=0.0):
    if epsilon > 0 and random.uniform(0, 1) < epsilon:
        action_index = random.randint(0, NUM_ACTIONS - 1)
    else:
        action_index = np.argmax(q_table[state])
    
    return ACTIONS[action_index], action_index

def ir_values_to_bins(values, thresholds=BIN_THRESHOLDS) -> tuple:
    """
    Transforms input values in list to binned values in a tuple based on given thresholds.
    
    Parameters:
    values (list): A list of numerical values to be binned.
    thresholds (list): A list of thresholds defining the bins.
    
    Returns:
    tuple: A tuple containing binned values.
    """
    print(f"New IR-Values are : {values}")
    state = []
    for value in values:
        bin_assigned = False
        for i, threshold in enumerate(thresholds):
            if value < threshold:
                state.append(i)
                bin_assigned = True
                break
        if not bin_assigned:
            state.append(IR_BINS - 1)  # Assign to the last bin if no threshold matched
    
    print(f"Binned to state: {state}")
    return tuple(state)

# Function that creates header for csv in directory
def create_csv_with_header(header_values, dir_str):
    with open(dir_str,'a') as f:
        for item in header_values:
            f.write(str(item))
            f.write(';')
        f.write('\n')

# Funciton that moves robot and returns new state
def play_robot_action(rob, action=None):
    # move robot
    print(f"Simulating action: {action}")
    move_robot(rob, action)

    # get new IR values and state
    ir_values = get_IR_values(rob)
    next_state = ir_values_to_bins(ir_values, thresholds=BIN_THRESHOLDS_HARDWARE)

    return next_state

# Function that retrieves IR-sensor values from robot
def get_IR_values(rob) -> list:
    ir_values = rob.read_irs()

    left_left_IR = ir_values[7]
    center_IR = ir_values[4]
    right_right_IR = ir_values[5]

    selected_values = [left_left_IR, center_IR, right_right_IR]

    return [round(value) for value in selected_values]

# Funciton that moves robot
def move_robot(rob, action):
    if action == 'forward':
        rob.move_blocking(FORWARD_SPEED_LEFT, FORWARD_SPEED_RIGHT, FORWARD_DURATION)
    elif action == 'right':
        rob.move_blocking(RIGHT_SPEED_LEFT, RIGHT_SPEED_RIGHT, RIGHT_DURATION)
    elif action == 'left':
        rob.move_blocking(LEFT_SPEED_LEFT, LEFT_SPEED_RIGHT, LEFT_DURATION)
       
    rob.sleep(0.1)  # block for _ seconds

# Function that takes the action and calculates reward
def simulate_robot_action(rob, action=None):

    # move robot and observe new state
    next_state = play_robot_action(rob, action)

    # Compute reward of action
    # Check if any of the sensors collided
    if COLLISION_STATE in next_state:
        reward = HIT_PENALTY
    else:
        reward = 1  # Default reward

    if action == 'forward' and reward != HIT_PENALTY:
        reward += FORWARD_REWARD
    
    # if falls of map, sensors are 0, then stop simulation
    if 0 in next_state:
        done = True
    else: done = False

    return next_state, reward, done



# Training function using Q-learning
def train_q_table(rob, run_name, q_table, q_table_path,results_path, num_episodes=200, max_steps=40, alpha=0.1, gamma=0.9, epsilon=0.1):
    # setup data file to store metrics while training
    create_csv_with_header(header_values=['run_name',
                                          'IR_BINS',
                                          'BIN_THRESHOLDS',
                                          'episode',
                                          'step',
                                          'reward',
                                          'action',
                                          'selected_values',
                                          'state'], 
                            directory= str(RESULT_DIR / results_path))
    
    for episode in range(num_episodes):
        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()
            print("Start simulation: ", episode)

        state = (1,1,1)
        done = False

        for step in range(max_steps):
            print("Episode: ", episode, "Step: ", step)

            # Choose an action, random by prob. epsilon, max, by prob 1-epsilon
            action, action_index = get_action(q_table, state, epsilon)

            # Take the action and observe the new state and reward
            new_state, reward, done = simulate_robot_action(rob, action)
            print(f"Got new reward {reward}")
            
            # Update the Q-value in Q-table
            max_future_q = max(q_table[new_state])
            current_q = q_table[state][action_index]
            q_table[state][action_index] = current_q + alpha * (reward + gamma * max_future_q - current_q)

            # Store result in csv.
            write_csv(data_values= [run_name,
                                    IR_BINS,
                                    BIN_THRESHOLDS,
                                    episode,
                                    step,
                                    reward,
                                    action,
                                    get_IR_values(rob),
                                    state],
                      dir_str=str(RESULT_DIR / results_path))
            
            # Transition to the new state
            state = new_state

            # Check collision with object, or maximum steps is reached, then stop simulation
            if reward == HIT_PENALTY or step >= max_steps-1: 
                done = True
                if isinstance(rob, SimulationRobobo):
                    rob.stop_simulation()

            if done:
                break

        # Save Q-table periodically
        if episode % 10 == 0:
            save_q_table(q_table, q_table_path)
    
    print_q_table(q_table)

    # Save the final Q-table
    save_q_table(q_table, q_table_path)

    # Training function using Q-learning
def play_q_table(rob, q_table):

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        print("Start simulation")

    # Initialize the episode
    state = (1,1,1)
    done = False

    while True:
        # Determine action by state from q-table
        action, _ = get_action(q_table, state)

        # Take the action and observe the new state
        new_state = play_robot_action(rob, action)
        
        # Transition to the new state
        state = new_state

        # Check if robot has collided, then stop simulation
        if COLLISION_STATE in state: 
            done = True
            if isinstance(rob, SimulationRobobo):
                rob.stop_simulation()

        if done:
            break

    print_q_table(q_table)