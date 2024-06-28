import pickle
import numpy as np
import cv2
import os
import itertools
import random

from data_files import FIGRURES_DIR

from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

from learning_machines import Training_Results

# Initiate Training_Results object
global TRAINING_RESULTS

# GLOBAL VARIABLES

# Define the possible actions
ACTIONS = ['left', 'forward', 'right']
NUM_ACTIONS = len(ACTIONS)

RED_OBJECT_THRESHOLD = 30
GREEN_OBJECT_THRESHOLD = 40
COLOR_BIN_THRESHOLDS = [1, 20, 45]

# Define Greenness Constans
GREEN_BINS = 4 # 0,1,2,3
GREEN_BIN_THRESHOLDS = [1, 20, 45]
GREEN_BIN_THRESHOLDS_HARDWARE = [5, 20, 40]

GREEN_LOWER_COLOR = np.array([36, 25, 25])
GREEN_HIGHER_COLOR = np.array([70, 255,255])

GREEN_LOWER_COLOR_HARDWARE = np.array([10, 60, 10])
GREEN_HIGHER_COLOR_HARDWARE = np.array([100, 250, 100])

# Define Redness Constans
RED_BINS = 4 # 0,1,2,3
RED_BIN_THRESHOLDS = [1, 20, 45]
RED_BIN_THRESHOLDS_HARDWARE = [5, 20, 40]

RED_LOWER_COLOR = np.array([120,0,0])
RED_HIGHER_COLOR = np.array([250,80,80])

RED_LOWER_COLOR_HARDWARE = np.array([70,30,30])
RED_HIGHER_COLOR_HARDWARE = np.array([250,50,50])

# Define COLOR Constans
COLOR_BINS = 4 # 0,1,2,3
COLOR_BIN_THRESHOLDS = [1, 20, 45]
COLOR_BIN_THRESHOLDS_HARDWARE = [5, 20, 40]

COLOR_DIRECTION_BINS = 4
COLOR_DIRECTIONS = [0,1,2,3]

# Define rewards of moving
OBJECT_FOUND_REWARD = 50
COLOR_REWARD = 10
DIRECTION_REWARD = 5

# Define global constants for movement settings
FORWARD_SPEED_LEFT = 50
FORWARD_SPEED_RIGHT = 50
FORWARD_DURATION = 400

RIGHT_SPEED_LEFT = 40
RIGHT_SPEED_RIGHT = -40
RIGHT_DURATION = 50

LEFT_SPEED_LEFT = -40
LEFT_SPEED_RIGHT = 40
LEFT_DURATION = 50

FORWARD_SPEED_LEFT_HDW = 100
FORWARD_SPEED_RIGHT_HDW = 100
FORWARD_DURATION_HDW = 300

RIGHT_SPEED_LEFT_HDW = 30
RIGHT_SPEED_RIGHT_HDW = -30
RIGHT_DURATION_HDW = 200

LEFT_SPEED_LEFT_HDW = -30
LEFT_SPEED_RIGHT_HDW = 30
LEFT_DURATION_HDW = 200

# Define global variables for the image dimensions and clipping
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_DEPTH = 3 #RGB
CLIP_HEIGHT = IMAGE_HEIGHT // 3
CLIP_WIDTH = IMAGE_WIDTH // 4

IMAGE_CENTER_Y = IMAGE_WIDTH // 2
IMAGE_OBJECT_HEIGHT = 80
IMAGE_OBJECT_WIDTH = 100
CENTER_TOP_WIDTH = 40
CENTER_BOT_WIDTH = 80

# Functions for loading and saving q-table
def load_q_table(q_table_path):
    with open(q_table_path, 'rb') as f:
        return pickle.load(f)
    
def save_q_table(q_table, q_table_path):
    with open(q_table_path, 'wb') as f:
        pickle.dump(q_table, f)

def clear_csv(results_path):
    with open(results_path, mode='w', newline='') as file:
        pass

# Initialize Q-table
def initialize_q_table(q_table_path, num_color_bins=COLOR_BINS, num_direction_bins=COLOR_DIRECTION_BINS, num_actions=NUM_ACTIONS):
    """Load the Q-table from a file if it exists; otherwise, initialize a new Q-table."""
    if os.path.exists(q_table_path):
        with open(q_table_path, 'rb') as f:
            q_table = pickle.load(f)
    else:
        q_table = {}
        for state in itertools.product([False,True], range(num_color_bins), range(num_direction_bins)):
            q_table[state] = [0.0 for _ in range(num_actions)]
            
        save_q_table(q_table, q_table_path)  # Save the new Q-table for the first time
    
    return q_table

# Visualize Q-table
def print_q_table(q_table, num_entries=10):
    print(f"{'State':<15} | {'Q-values (left, forward, right)':<30}")
    print("-" * 50)
    for i, (state, q_values) in enumerate(q_table.items()):
        print(f"{str(state):<15} | {[round(i,2) for i in q_values]}")
        if i >= num_entries - 1:
            break

# Function that writes values into csv in directory
def write_csv(data_values, dir_str):
    with open(dir_str,'a') as f:
                for item in data_values:
                    f.write(str(item))
                    f.write(';')
                f.write('\n')

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

# Funciton that moves robot
def move_robot(rob, action):
    if action == 'forward':
        rob.move_blocking(FORWARD_SPEED_LEFT, FORWARD_SPEED_RIGHT, FORWARD_DURATION)
    elif action == 'right':
        rob.move_blocking(RIGHT_SPEED_LEFT, RIGHT_SPEED_RIGHT, RIGHT_DURATION)
    elif action == 'left':
        rob.move_blocking(LEFT_SPEED_LEFT, LEFT_SPEED_RIGHT, LEFT_DURATION)
       
    rob.sleep(0.1)  # block for _ seconds

# Funciton that moves robot
def move_robot_hardware(rob, action):
    if action == 'forward':
        rob.move_blocking(FORWARD_SPEED_LEFT_HDW, FORWARD_SPEED_RIGHT_HDW, FORWARD_DURATION_HDW)
        #rob.sleep(0.05)
    elif action == 'right':
        rob.move_blocking(RIGHT_SPEED_LEFT_HDW, RIGHT_SPEED_RIGHT_HDW, RIGHT_DURATION_HDW)
        #rob.sleep(0.05)
    elif action == 'left':
        rob.move_blocking(LEFT_SPEED_LEFT_HDW, LEFT_SPEED_RIGHT_HDW, LEFT_DURATION_HDW)
        #rob.sleep(0.05)
       
    rob.sleep(0.1)  # block for _ seconds

# Function that returns image for state setting
def get_state_img(rob: IRobobo, dir_str):
    # retrieve current view of camera
    image = rob.get_image_front()

    # clip the upper part of the image
    clipped_image = image[CLIP_HEIGHT:, :]
    cv2.imwrite(dir_str, clipped_image) # store image for testing reasons

    return clipped_image

# Function that calculates 'colorness' in image
def get_img_color_per(image) -> int:
    # Calculate percentage 'color' pixels
    color_pixel_count = np.count_nonzero(image)
    total_pixel_count = image.size
    colorness_percentage = (color_pixel_count / total_pixel_count) * 100

    return int(colorness_percentage)

# Return section with highest number of color pixels
def img_color_direction(sections) -> int:
    color_pixel_counts = [np.count_nonzero(section) for section in sections]

    # Return max pixel section if non-zero
    if np.any(color_pixel_counts):
        return np.argmax(color_pixel_counts)+1
    else:
        return 0

# Functiont that retrieves colorness and computes colorness bin
def get_state_color_sq(image):
    # Split the mask into vertical sections
    height, width = image.shape

    center_left_y = (width//2) - (IMAGE_OBJECT_WIDTH//2)
    center_right_y = (width//2) + (IMAGE_OBJECT_WIDTH//2)

    sections = [
        image[:, :center_left_y],
        image[:, center_left_y:center_right_y],
        image[:, center_right_y:]
    ]

    image_center = sections[1]
    cv2.imwrite(str(FIGRURES_DIR / "center_view.png"), image_center) # store image for testing reasons
    
    center_color_per = get_img_color_per(image_center)
    center_color_bin = value_to_bin(center_color_per, thresholds= COLOR_BIN_THRESHOLDS)
    color_direction = img_color_direction(sections)

    print(f"The % of color in center is {center_color_per} ({center_color_bin}), in direction {color_direction}")

    return center_color_bin, color_direction

# Function that retrieves colorness and computes colorness bin
def get_state_color_tri(image):
    # Split the mask into vertical sections
    height, width = image.shape

    center_top_left = [(width // 2) - (CENTER_TOP_WIDTH // 2),0]
    center_top_right = [(width // 2) + (CENTER_TOP_WIDTH // 2),0]
    center_bot_left = [(width // 2) - (CENTER_BOT_WIDTH // 2), height-1]
    center_bot_right = [(width // 2) + (CENTER_BOT_WIDTH // 2), height-1]

    image_top_left = [0 , 0]
    image_top_right = [width , 0]
    image_bot_left = [0 , height-1]
    image_bot_right  = [width , height-1]
    
    # Create a mask for the triangular center section
    center_mask = np.zeros_like(image, dtype=np.uint8)
    pts = np.array([center_top_left, center_bot_left,center_bot_right,center_top_right], dtype=np.int32)
    cv2.fillPoly(center_mask, [pts], (255))

    # Create masks for the left and right sections
    left_mask = np.zeros_like(image, dtype=np.uint8)
    left_pts = np.array([image_top_left,image_bot_left,center_bot_left,center_top_left], dtype=np.int32)
    cv2.fillPoly(left_mask, [left_pts], (255))

    right_mask = np.zeros_like(image, dtype=np.uint8)
    right_pts = np.array([center_top_right,center_bot_right,image_bot_right,image_top_right], dtype=np.int32)
    cv2.fillPoly(right_mask, [right_pts], (255))
    
    # Apply the mask to get the triangular center section
    image_center = cv2.bitwise_and(image, center_mask)

    # Save the center image for testing reasons
    cv2.imwrite(str(FIGRURES_DIR / "center_view.png"), image_center)
    
    # Apply the masks to get the left and right sections
    image_left = cv2.bitwise_and(image, left_mask)
    # Save the center image for testing reasons
    cv2.imwrite(str(FIGRURES_DIR / "left_view.png"), image_left)
    
    image_right = cv2.bitwise_and(image, right_mask)
    # Save the center image for testing reasons
    cv2.imwrite(str(FIGRURES_DIR / "right_view.png"), image_right)
    
    sections = [image_left, image_center, image_right]

    center_color_per = get_img_color_per(image_center)
    center_color_bin = value_to_bin(center_color_per, thresholds= COLOR_BIN_THRESHOLDS)
    color_direction = img_color_direction(sections)

    try:
        TRAINING_RESULTS.steps['color_direction'] = color_direction
        TRAINING_RESULTS.steps['center_color_per'] = center_color_per
        TRAINING_RESULTS.steps['center_color_bin'] = center_color_bin
    except: pass

    print(f"The % of color in center is {center_color_per} ({center_color_bin}), in direction {color_direction}")

    return center_color_bin, color_direction

# function that returns color filtered image
def filter_red(img, lower_color=RED_LOWER_COLOR, higher_color=RED_HIGHER_COLOR):
    mask_red = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), lower_color, higher_color)
    cv2.imwrite(str(FIGRURES_DIR / "red_filter.png"), mask_red) # store image for testing reasons

    return mask_red


def filter_green(img, lower_color=GREEN_LOWER_COLOR, higher_color=GREEN_HIGHER_COLOR):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask_green = cv2.inRange(hsv, lower_color, higher_color)
        cv2.imwrite(str(FIGRURES_DIR / "green_filter.png"), mask_green) # store image for testing reasons

        return mask_green

def get_object_view_per(image):
    # clip image to object detection size
    height, width = image.shape
    
    object_view = image[height-IMAGE_OBJECT_HEIGHT:, (width//2)-(IMAGE_OBJECT_WIDTH//2):(width//2)+(IMAGE_OBJECT_WIDTH//2)]
    cv2.imwrite(str(FIGRURES_DIR / "object_view.png"), object_view) # store image for testing reasons

    colorness = get_img_color_per(object_view)
    print(f"colorness of object view is : {colorness}")

    return colorness

# Function that gets state
def get_state(rob, color):
    # get clipped image
    state_img = get_state_img(rob, str(FIGRURES_DIR / "state_image_test1.png"))
    
    # filter for color and get state components
    state_img_filter = None
    if color == 'red':
        state_img_filter = filter_red(state_img, lower_color=RED_LOWER_COLOR, higher_color=RED_HIGHER_COLOR)
        center_color_bin, color_direction = get_state_color_tri(state_img_filter)    # use triangle view
    
    if color == 'green':
        state_img_filter = filter_green(state_img, lower_color=GREEN_LOWER_COLOR, higher_color=GREEN_HIGHER_COLOR)
        center_color_bin, color_direction = get_state_color_sq(state_img_filter)     # use square view
    
    object_view_per = get_object_view_per(state_img_filter)
    try:
        TRAINING_RESULTS.steps['object_view_per'] = object_view_per
    except: pass   

    found_object = False
    if color=='red' and object_view_per>=RED_OBJECT_THRESHOLD:
        found_object = True
    elif color=='green' and object_view_per>=GREEN_OBJECT_THRESHOLD:
        found_object = True

    return (found_object, center_color_bin, color_direction)

# Funciton that computes reward on the action and new_state:
def compute_reward(state, action, new_state):       # state: [Object_Found, % Center Colorness, Direction, Color]
    reward = 0

    # if object has been detected
    if new_state[0]: reward += OBJECT_FOUND_REWARD

    # if increased colorness
    if new_state[1]>state[1]: reward += COLOR_REWARD
    
    # if de-creased colorness
    if new_state[1]<state[1] and action != "forward" : reward -= 2*COLOR_REWARD

    # if increased direction
    if new_state[2] ==2 : reward += DIRECTION_REWARD
    if state[2] != 2 and new_state[2] ==2: reward += 2*DIRECTION_REWARD
    if state[2] == 0 and new_state[2] != 0: reward += DIRECTION_REWARD
    
    # if de-creased direction
    if state[2]==2 and new_state[2] != 2: reward -= 3       # not to hard penalty when driving forward and object is far away
    if state[2]!= 0 and new_state[2] == 0: reward -= 30

    return reward


# Function that retrieves action from q_table
# if epsilon is given, retrieve with creativity, else retrive argmax
def get_action(q_table, state, epsilon=0.0):
    if epsilon > 0 and random.uniform(0, 1) < epsilon:
        action_index = random.randint(0, NUM_ACTIONS - 1)
    else:
        action_index = np.argmax(q_table[state])
    
    return ACTIONS[action_index], action_index

# Function that moves robot and returns new state
def play_robot_action(rob, action, color):
    # move robot
    move_robot(rob, action)

    #return new state
    return get_state(rob, color)

# Function that moves robot and returns new state
def play_robot_action_hardware(rob, action, color):
    # move robot
    move_robot_hardware(rob, action)

    #return new state
    return get_state(rob, color)

# Training function using Q-learning
def train_q_table(rob, color, run_name, q_table, q_table_path,results_path, num_episodes=200, max_steps=40, alpha=0.1, gamma=0.9, epsilon=0.1):
    global TRAINING_RESULTS
    # Initialize Training Object and CSV to store results
    clear_csv(results_path)

    TRAINING_RESULTS = Training_Results.Training_Results(run_name=run_name,
                                        color=color,
                                        color_bin_thresholds=COLOR_BIN_THRESHOLDS,
                                        num_episodes=num_episodes,
                                        max_steps=max_steps,
                                        alpha=alpha,
                                        gamma=gamma,
                                        epsilon=epsilon)
    
    TRAINING_RESULTS.create_csv_with_header(results_path)

    for episode in range(num_episodes):
        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()
            print("------ Start simulation: ", episode)
            print("\n")

        # Move phone to start view
        rob.set_phone_tilt_blocking(109, 60)

        # Build up state
        state = get_state(rob, color)
        done = False

        for step in range(max_steps):
            print("Episode: ", episode, "Step: ", step)
            TRAINING_RESULTS.steps['episode'] = episode
            TRAINING_RESULTS.steps['step'] = step
            TRAINING_RESULTS.steps['state'] = state

            # Choose an action, random by prob. epsilon, max, by prob 1-epsilon
            action, action_index = get_action(q_table, state, epsilon)
            TRAINING_RESULTS.steps['action'] = action

            # Take the action and observe the new state:[Object, %Center Color, Direction most Color]
            new_state = play_robot_action(rob, action, color)

            # compute reward
            reward = compute_reward(state, action, new_state)

            print(f"Moved from state {state}")
            print(f"        to state {new_state}")
            print(f"              by {action}")
            print(f"                            reward {reward}")

            TRAINING_RESULTS.steps['new_state'] = new_state
            TRAINING_RESULTS.steps['reward'] = reward

            # Update the Q-value in Q-table
            max_future_q = max(q_table[new_state])
            current_q = q_table[state][action_index]
            q_table[state][action_index] = current_q + alpha * (reward + gamma * max_future_q - current_q)

            
            # If found the object, stop episode
            if new_state[0] == True:
                rob.talk(f"Found object!")
                done = True
                print(f"-----------------Found {color} -----------------\n")
                print(f"------------------- END EPISODE {episode} --------------------")

            
            # Store result in csv.
            TRAINING_RESULTS.write_line_to_csv(results_path)

            # Transition to the new state
            state = new_state

            if done:
                break
        
        if done:
            print(f"-----------------Collecting {color} -----------------\n")
            rob.move_blocking(40,40,400)
            rob.move_blocking(-40,-40,800)

        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()

        # Save Q-table periodically
        if episode % 10 == 0:
            save_q_table(q_table, q_table_path)
    
    print_q_table(q_table, num_entries=50)

    # Save the final Q-table
    save_q_table(q_table, q_table_path)


# Training function using Q-learning
def play_q_table(rob, q_table, epsilon, hardware_flag=False):

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        print("Start simulation")
    
    # Move phone to start view
    rob.set_phone_tilt_blocking(109, 20)

    
    # Build up state for RED
    print("-----------------Searching RED -----------------\n")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.RED)
    state = get_state(rob, color='red')    
    done = False

    while True:
        # Determine action by state from q-table
        action, action_index = get_action(q_table, state, epsilon)

        # Take the action and observe the new state
        if hardware_flag:
            new_state = play_robot_action_hardware(rob, action, color='red')

        else:
            new_state = play_robot_action(rob, action, color='red')

        print(f"Moved from state {state}")
        print(f"        to state {new_state}")
        print(f"              by {action}")

        if new_state[0] == True:
            print("-----------------Found RED -----------------\n")
            rob.talk(f"Found red object!")
            done = True

        # Transition to the new state
        state = new_state

        if done:
            break
    
    # move into object
    rob.move_blocking(40,40,1800)

    
    # Build up state for GREEN
    print("-----------------Searching GREEN-----------------\n")
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)
    state = get_state(rob, color='green')    
    done = False

    while True:
        # Determine action by state from q-table
        action, action_index = get_action(q_table, state, epsilon)

        # Take the action and observe the new state
        if hardware_flag:
            new_state = play_robot_action_hardware(rob, action, color='green')

        else:
            new_state = play_robot_action(rob, action, color='green')

        print(f"Moved from state {state}")
        print(f"        to state {new_state}")
        print(f"              by {action}")

        if new_state[0] == True:
            print("------------------ Found GREEN -----------------\n")
            rob.talk(f"Found green object!")
            rob.talk("JAJAJAJJAJAJJAJA")
            rob.set_led(LedId.FRONTCENTER, LedColor.MAGENTA)
            done = True

        # Transition to the new state
        state = new_state

        if done:
            break

    
    # move object
    rob.move_blocking(40,40,1600)
    rob.move_blocking(-40,-40,1800)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
        print("Stop simulation")

# Function to test actions
def test_robo(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        print("Start simulation")
    
    rob.set_phone_tilt_blocking(109, 60)
    
    # rob.move_blocking(50, -10, 920) # 90 Degrees Right
    # rob.move_blocking(-10, 50, 460) # 45 Degrees Left
    # rob.move_blocking(60, 60, 2300) # 1 Grid forward
    
    rob.move_blocking(-60, -60, 700)
    center_IR = rob.read_irs()[4]
    print(f"The center low IR is {center_IR}")

    print("Forwards")
    rob.move_blocking(60, 60, 2300)
    center_IR = rob.read_irs()[4]
    print(f"The center low IR is {center_IR}")
    rob.move_blocking(60, 60, 2300)
    center_IR = rob.read_irs()[4]
    print(f"The center low IR is {center_IR}")
    rob.move_blocking(60, 60, 2300)
    center_IR = rob.read_irs()[4]
    print(f"The center low IR is {center_IR}")
    rob.move_blocking(60, 60, 2300)
    center_IR = rob.read_irs()[4]
    print(f"The center low IR is {center_IR}")

    print("Backwards")
    rob.move_blocking(-60, -60, 700)
    center_IR = rob.read_irs()[4]
    print(f"The center low IR is {center_IR}")
    rob.move_blocking(-60, -60, 700)
    center_IR = rob.read_irs()[4]
    print(f"The center low IR is {center_IR}")
    rob.move_blocking(-60, -60, 700)
    center_IR = rob.read_irs()[4]
    print(f"The center low IR is {center_IR}")
    rob.move_blocking(-60, -60, 700)
    center_IR = rob.read_irs()[4]
    print(f"The center low IR is {center_IR}")

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()