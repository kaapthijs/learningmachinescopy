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
ACTIONS = ['forward', 'left', 'right']
NUM_ACTIONS = len(ACTIONS)

# Define number of bins where sensor falls in
NUM_BINS = 5    # sensor value could be 0,1,2,3
BIN_THRESHOLDS = [4,7,10,15]

# Functions for loading and saving q-table
def load_q_table(file_path='q_table.pkl'):
    with open(str(RESULT_DIR / file_path), 'rb') as f:
        return pickle.load(f)
    
def save_q_table(q_table, file_path='q_table.pkl'):
    with open(str(RESULT_DIR / file_path), 'wb') as f:
        pickle.dump(q_table, f)


# Initialize Q-table
def initialize_q_table(file_path='q_table.pkl', num_bins=NUM_BINS, num_sensors=NUM_SENSORS, num_actions=NUM_ACTIONS):
    """Load the Q-table from a file if it exists; otherwise, initialize a new Q-table."""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            q_table = pickle.load(f)
    else:
        q_table = {}
        for state in itertools.product(range(num_bins), repeat=num_sensors):
            q_table[state] = [0.0 for _ in range(num_actions)]
            
        save_q_table(q_table, file_path)  # Save the new Q-table for the first time
    
    return q_table

# Visualize Q-table
def print_q_table(q_table, num_entries=10):
    print(f"{'State':<15} | {'Q-values (forward, left, right)':<30}")
    print("-" * 50)
    for i, (state, q_values) in enumerate(q_table.items()):
        print(f"{str(state):<15} | {q_values}")
        if i >= num_entries - 1:
            break

def transform_ir_values(values, thresholds=BIN_THRESHOLDS):
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
            state.append(NUM_BINS - 1)  # Assign to the last bin if no threshold matched
    
    print(f"Binned to state: {state}")
    return tuple(state)

# Function that takes the action and calculates reward
def simulate_robot_action(rob, action=None):

    print(f"Simulating action: {action}")
    if action == 'forward':
        rob.move_blocking(50, 50, 1000)
    elif action == 'left':
        rob.move_blocking(50, -10, 500)
    elif action == 'right':
        rob.move_blocking(-10, 50, 500)
       
    rob.sleep(0.1)

    ir_values = rob.read_irs()
    selected_values = [ir_values[7], ir_values[4], ir_values[5]]
    selected_values = [round(value) for value in selected_values]

    next_state = transform_ir_values(selected_values)

    # Compute reward of action
    # Check if any of the sensors exceed their respective thresholds
    #if any((value > threshold1 and idx < 2) or (value > threshold2 and idx < 4) or (value > threshold_rest and idx >= 4) for idx, value in enumerate(selected_values)):
    if any(value==NUM_BINS-1 for value in next_state):    
        reward = -50  # Penalty for hitting an object
    else:
        reward = 1  # Default reward

    if action == 'forward' and reward != -50:
        forward_reward = 10
        reward += forward_reward
    
    # if falls of map, sensors are 0, then stop simulation
    if next_state == (0,) * NUM_SENSORS:
        done = True
    else: done = False

    return next_state, reward, done

# Training function using Q-learning
def train_q_table(rob, q_table, num_episodes=200, max_steps=40, alpha=0.1, gamma=0.9, epsilon=0.1):
    for episode in range(num_episodes):
        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()
            print("Start simulation: ", episode)

        # Initialize the episode
        ir_values = rob.read_irs()
        selected_values = [5,5,5] # avoid starting with [inf, inf, inf]
    
        state = (1,1,1)
        done = False

        for step in range(max_steps):
            print("Episode: ", episode, "Step: ", step)

            # Choose an action, random by prob. epsilon, max, by prob 1-epsilon
            if random.uniform(0, 1) < epsilon:
                action_index = random.randint(0, NUM_ACTIONS - 1)
            else:
                action_index = np.argmax(q_table[state])

            action = ACTIONS[action_index]

            # Take the action and observe the new state and reward
            new_state, reward, done = simulate_robot_action(rob, action)
            print(f"Got new reward {reward}")
            
            # Update the Q-value
            max_future_q = max(q_table[new_state])
            current_q = q_table[state][action_index]
            q_table[state][action_index] = current_q + alpha * (reward + gamma * max_future_q - current_q)

            # Transition to the new state
            state = new_state

            if reward == -50 or step >= max_steps: 
                done = True
                if isinstance(rob, SimulationRobobo):
                    rob.stop_simulation()

            if done:
                break

        # Optionally save Q-table periodically
        if episode % 10 == 0:
            save_q_table(q_table)
    
    print_q_table(q_table)

    # Save the final Q-table
    save_q_table(q_table)