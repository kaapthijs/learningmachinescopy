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
IR_BINS = 4    # sensor value could be 0,1,2,3
BIN_THRESHOLDS = [4,10,25]
BIN_THRESHOLDS_HARDWARE = [-1,15,100]

# Define Greenness Constans
GREEN_BINS = 3 # 0,1,2
GREEN_THRESHOLDS = [60,80]
GREEN_THRESHOLDS_HARDWARE = [5,10,15]

GREEN_LOWER = np.array([0, 160, 0])
GREEN_HIGHER = np.array([140, 255, 140])

GREEN_LOWER_HARDWARE = np.array([0, 160, 0])
GREEN_HIGHER_HARDWARE = np.array([140, 255, 140])

GREEN_DIRECTION_BINS = 5
GREEN_DIRECTION_THRESHOLDS = [0,1,2,3,4]

# Define rewards of moving
FORWARD_REWARD = 11
HIT_PENALTY = -50
COLLISION_STATE = IR_BINS-1

# Define global constants for movement settings
FORWARD_SPEED_LEFT = 50
FORWARD_SPEED_RIGHT = 50
FORWARD_DURATION = 300

RIGHT_SPEED_LEFT = 60
RIGHT_SPEED_RIGHT = -60
RIGHT_DURATION = 300

LEFT_SPEED_LEFT = -60
LEFT_SPEED_RIGHT = 60
LEFT_DURATION = 300

# Define global variables for the image dimensions and clipping
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_DEPTH = 3 #RGB
CLIP_HEIGHT = IMAGE_HEIGHT // 2

# Functions for loading and saving q-table
def load_q_table(q_table_path):
    with open(str(RESULT_DIR / q_table_path), 'rb') as f:
        return pickle.load(f)
    
def save_q_table(q_table, q_table_path):
    with open(str(RESULT_DIR / q_table_path), 'wb') as f:
        pickle.dump(q_table, f)

# Initialize Q-table
def initialize_q_table(q_table_path, num_ir_bins=IR_BINS, num_green_bins=GREEN_BINS, num_direction_bins=GREEN_DIRECTION_BINS, num_actions=NUM_ACTIONS):
    """Load the Q-table from a file if it exists; otherwise, initialize a new Q-table."""
    if os.path.exists(q_table_path):
        with open(q_table_path, 'rb') as f:
            q_table = pickle.load(f)
    else:
        q_table = {}
        for state in itertools.product(range(1,num_ir_bins), range(num_green_bins), range(num_direction_bins)):
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

def ir_values_to_bins(values, thresholds=BIN_THRESHOLDS):
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

# Function that transforms value to bin value
def value_to_bin(value, thresholds):
    for i, threshold in enumerate(thresholds):
        if value < threshold:
            return i
    return len(thresholds)

# Function that creates header for csv in directory
def create_csv_with_header(header_values, dir_str):
    with open(dir_str,'a') as f:
        for item in header_values:
            f.write(str(item))
            f.write(';')
        f.write('\n')

# Function that moves robot and returns new state
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

    selected_values = [center_IR]

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

# Function that returns image for state setting
def get_state_img(rob: IRobobo, dir_str):
    # retrieve current view of camera
    image = rob.get_image_front()

    # clip the upper part of the image
    clipped_image = image[CLIP_HEIGHT:, :]
    cv2.imwrite(dir_str, clipped_image) # store image for testing reasons

    # for test move to:
    if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()

    return clipped_image

# Function that calculates 'greenness' in image
def calculate_img_greenness(image) -> int:
    # Filter green color
    mask_green = cv2.inRange(image, GREEN_LOWER, GREEN_HIGHER)
    cv2.imwrite(str(FIGRURES_DIR / "green_filter_test.png"), mask_green)

    # Calculate percentage 'green' pixels
    green_pixel_count = np.count_nonzero(mask_green)
    total_pixel_count = mask_green.size
    greenness_percentage = (green_pixel_count / total_pixel_count) * 100

    return int(greenness_percentage)

def img_greenness_direction(image) -> int:
    # Filter green color
    mask_green = cv2.inRange(image, GREEN_LOWER, GREEN_HIGHER)

    # Split the mask into five vertical sections
    height, width = mask_green.shape
    section_width = width // 5

    sections = [
        mask_green[:, :section_width],
        mask_green[:, section_width:2*section_width],
        mask_green[:, 2*section_width:3*section_width],
        mask_green[:, 3*section_width:4*section_width],
        mask_green[:, 4*section_width:]
    ]

    # Count the number of green pixels in each section
    green_pixel_counts = [np.count_nonzero(section) for section in sections]

    # Determine which section has the most green pixels
    max_index = np.argmax(green_pixel_counts)
    green_section = [0,1,2,3,4]

    return green_section[max_index]

# Functiont that retrieves greenness and computes greenness bin
def get_state_greenness(image):
    # get greenness value %
    greenness = calculate_img_greenness(image)
    green_direction = img_greenness_direction(image)

    print(f"The % of greenness is {greenness}")
    print(f"The greennes direction is {green_direction}")
    
    # transform greenness values into bin
    greenness = value_to_bin(greenness, thresholds= GREEN_THRESHOLDS)

    return greenness, green_direction


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
    """
    create_csv_with_header(header_values=['run_name',
                                          'IR_BINS',
                                          'BIN_THRESHOLDS',
                                          'episode',
                                          'step',
                                          'reward',
                                          'action',
                                          'selected_values',
                                          'state'], 
                            dir_str= str(RESULT_DIR / results_path))
    """
    for episode in range(num_episodes):
        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()
            print("Start simulation: ", episode)

        # Move phone to start view
        rob.set_phone_tilt_blocking(100, 20)

        # rotate robot to test view
        rob.move_blocking(-60,60, 1000)
        rob.sleep(0.5)

        # Build up state components
        state_img = get_state_img(rob, str(FIGRURES_DIR / "state_image_test1.png"))
        state_greenness, greenness_direction = get_state_greenness(state_img)

        state = (1,state_greenness, greenness_direction)

        print(f"State became {state}")
        """
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
    save_q_table(q_table, q_table_path)"""

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