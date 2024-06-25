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
# Define number of sensors used
NUM_SENSORS = 1

# Define the possible actions
ACTIONS = ['left', 'forward', 'right']
NUM_ACTIONS = len(ACTIONS)

# Define number of InfraRed bins where sensor falls in
IR_BINS = 4    # sensor value could be 0,1,2,3
IR_BIN_THRESHOLDS = [4,7,550]
IR_BIN_THRESHOLDS_HARDWARE = [-1, 20, 60]

# Define Greenness Constans
GREEN_BINS = 4 # 0,1,2,3
GREEN_BIN_THRESHOLDS = [1, 20, 45]
GREEN_BIN_THRESHOLDS_HARDWARE = [5, 20, 40]

GREEN_LOWER_COLOR = np.array([0, 160, 0])
GREEN_HIGHER_COLOR = np.array([140, 255, 140])

GREEN_LOWER_COLOR_HARDWARE = np.array([30, 90, 30])
GREEN_HIGHER_COLOR_HARDWARE = np.array([85, 237, 85])

GREEN_DIRECTION_BINS = 5
GREEN_DIRECTIONS = [1,2,3,4,5]

# Define Redness Constans
RED_BINS = 4 # 0,1,2,3
RED_BIN_THRESHOLDS = [1, 20, 45]
RED_BIN_THRESHOLDS_HARDWARE = [5, 20, 40]

RED_LOWER_COLOR = np.array([120,0,0])
RED_HIGHER_COLOR = np.array([250,80,80])

RED_LOWER_COLOR_HARDWARE = np.array([120,0,0])
RED_HIGHER_COLOR_HARDWARE = np.array([250,100,80])

RED_DIRECTION_BINS = 4
RED_DIRECTIONS = [0,1,2,3]

# Define rewards of moving
ALMOST_HIT_PENALTY = -25
HIT_PENALTY = -50
ALMOST_COLLISION_STATE = IR_BINS-2
COLLISION_STATE = IR_BINS-1
FOOD_HIT_STATE = GREEN_BINS-1
FOOD_REWARD = 50
GREEN_REWARD = 15
FORWARD_REWARD = 8 # encourage getting closer to get in vision range of objects
LEFT_REWARD = 25 # when hitting the wall straight up receive a left reward

# Define global constants for movement settings
FORWARD_SPEED_LEFT = 50
FORWARD_SPEED_RIGHT = 50
FORWARD_DURATION = 300

RIGHT_SPEED_LEFT = 50
RIGHT_SPEED_RIGHT = -10
RIGHT_DURATION = 460

LEFT_SPEED_LEFT = -10
LEFT_SPEED_RIGHT = 50
LEFT_DURATION = 460

FORWARD_SPEED_LEFT_HDW = 100
FORWARD_SPEED_RIGHT_HDW = 100
FORWARD_DURATION_HDW = 300

RIGHT_SPEED_LEFT_HDW = 45
RIGHT_SPEED_RIGHT_HDW = -40
RIGHT_DURATION_HDW = 200

LEFT_SPEED_LEFT_HDW = -40
LEFT_SPEED_RIGHT_HDW = 45
LEFT_DURATION_HDW = 200

# Define global variables for the image dimensions and clipping
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_DEPTH = 3 #RGB
CLIP_HEIGHT = IMAGE_HEIGHT // 3
CLIP_WIDTH = IMAGE_WIDTH // 4

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
    print(f"{'State':<15} | {'Q-values (left, forward, right)':<30}")
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

def ir_values_to_bins(values, thresholds=IR_BIN_THRESHOLDS):
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

# Function that retrieves IR-sensor values from robot
def get_IR_values(rob) -> list:
    ir_values = rob.read_irs()

    #left_left_IR = ir_values[7]
    center_IR = ir_values[4]
    #right_right_IR = ir_values[5]

    TRAINING_RESULTS.steps['center_IR'] = center_IR

    return round(center_IR)

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
    clipped_image = image[CLIP_HEIGHT:, CLIP_WIDTH:-CLIP_WIDTH]
    cv2.imwrite(dir_str, clipped_image) # store image for testing reasons

    return clipped_image

# Function that calculates 'colourness' in image
def calculate_img_colourness(image) -> int:
    # Calculate percentage 'green' pixels
    green_pixel_count = np.count_nonzero(image)
    total_pixel_count = image.size
    greenness_percentage = (green_pixel_count / total_pixel_count) * 100

    return int(greenness_percentage)

def img_greenness_direction(image) -> int:
    # Split the mask into five vertical sections
    height, width = image.shape
    section_width = width // GREEN_DIRECTION_BINS

    sections = [
        image[:, :section_width],
        image[:, section_width:2*section_width],
        image[:, 2*section_width:3*section_width],
        image[:, 3*section_width:4*section_width],
        image[:, 4*section_width:]
    ]

    # Count the number of green pixels in each section
    green_pixel_counts = [np.count_nonzero(section) for section in sections]
    TRAINING_RESULTS.steps['green_pixels'] = green_pixel_counts

    # Determine which section has the most green pixels
    max_index = np.argmax(green_pixel_counts)
    
    if all(value == 0 for value in green_pixel_counts):
        return 0
    else:
        return max_index

def img_redness_direction(image) -> int:
    # Split the mask into five vertical sections
    height, width = image.shape
    section_width = width // RED_DIRECTION_BINS

    sections = [
        image[:, :section_width],
        image[:, section_width:3*section_width],
        image[:, 3*section_width:]
    ]

    # Count the number of red pixels in each section
    red_pixel_counts = [np.count_nonzero(section) for section in sections]
    TRAINING_RESULTS.steps['red_pixels'] = red_pixel_counts

    # Determine which section has the most red pixels
    max_index = np.argmax(red_pixel_counts)
    
    if all(value == 0 for value in red_pixel_counts):
        return 0
    else:
        return max_index+1

# Functiont that retrieves greenness and computes greenness bin
def get_state_greenness(image, lower_color=GREEN_LOWER_COLOR, higher_color=GREEN_HIGHER_COLOR):
    # get greenness value % from center view
    # Split the mask into vertical sections
    # Filter green color
    mask_green = cv2.inRange(image, lower_color, higher_color)
    cv2.imwrite(str(FIGRURES_DIR / "green_filter.png"), mask_green) # store image for testing reasons
    

    # Split the mask into vertical sections
    height, width = mask_green.shape
    section_width = width // GREEN_DIRECTION_BINS

    image_centre = mask_green[:, section_width:3*section_width]
    cv2.imwrite(str(FIGRURES_DIR / "green_filter_center.png"), image_centre) # store image for testing reasons
    
    greenness = calculate_img_colourness(image_centre)

    green_direction = img_greenness_direction(mask_green)

    print(f"The % of greenness in centre is {greenness}, in direction {green_direction}")
    
    # transform greenness values into bin
    greenness = value_to_bin(greenness, thresholds= GREEN_BIN_THRESHOLDS)

    return greenness, green_direction

# Functiont that retrieves redness and computes redness bin
def get_state_redness(image, lower_color=RED_LOWER_COLOR, higher_color=RED_HIGHER_COLOR):
    # get redness value % from center view
    # Split the mask into vertical sections
    # Filter red color
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_red = cv2.inRange(image, lower_color, higher_color)
    cv2.imwrite(str(FIGRURES_DIR / "red_filter.png"), mask_red) # store image for testing reasons
    

    # Split the mask into vertical sections
    height, width = mask_red.shape
    section_width = width // RED_DIRECTION_BINS

    image_centre = mask_red[:, section_width:3*section_width]
    cv2.imwrite(str(FIGRURES_DIR / "red_filter_center.png"), image_centre) # store image for testing reasons
    
    redness = calculate_img_colourness(image_centre)

    red_direction = img_redness_direction(mask_red)

    print(f"The % of redness in centre is {redness}, in direction {red_direction}")
    
    # transform redness values into bin
    redness = value_to_bin(redness, thresholds= RED_BIN_THRESHOLDS)

    return redness, red_direction



# Function that gets state
def get_state(rob, thresholds, lower_color=GREEN_LOWER_COLOR, higher_color=GREEN_HIGHER_COLOR):

    state_ir = value_to_bin(get_IR_values(rob), thresholds)
    state_img = get_state_img(rob, str(FIGRURES_DIR / "state_image_test1.png"))
    state_greenness, green_direction = get_state_greenness(state_img, lower_color, higher_color)

    TRAINING_RESULTS.steps['center_bin'] = state_ir
    TRAINING_RESULTS.steps['greenness_bin'] = state_greenness
    TRAINING_RESULTS.steps['green_direction'] = green_direction

    return (state_ir,state_greenness, green_direction)

# Function that retrieves action from q_table
# if epsilon is given, retrieve with creativity, else retrive argmax
def get_action(q_table, state, epsilon=0.0):
    if epsilon > 0 and random.uniform(0, 1) < epsilon:
        action_index = random.randint(0, NUM_ACTIONS - 1)
    else:
        action_index = np.argmax(q_table[state])
    
    return ACTIONS[action_index], action_index

# Function that moves robot and returns new state
def play_robot_action(rob, action=None, thresholds=IR_BIN_THRESHOLDS):
    # move robot
    move_robot(rob, action)

    #return new state
    return get_state(rob, thresholds)

# Function that moves robot and returns new state
def play_robot_action_hardware(rob, action=None, thresholds=IR_BIN_THRESHOLDS):
    # move robot
    move_robot_hardware(rob, action)

    #return new state
    return get_state(rob, thresholds, GREEN_LOWER_COLOR_HARDWARE, GREEN_HIGHER_COLOR_HARDWARE)


# Function that takes the action and calculates reward
def simulate_robot_action(rob, action=None):

    # move robot and observe new state
    next_state = play_robot_action(rob, action)
    reward = 0
    done=False

    # Compute reward of action, state:(IR_Distance, %Greenness, Dir.Greenness)
    """
    if next_state[0]==COLLISION_STATE and next_state[1]==FOOD_HIT_STATE:   # Check if collected food: Close distance AND Green
        reward += FOOD_REWARD
        if action == 'forward':
            reward += FOOD_REWARD + GREEN_REWARD + FORWARD_REWARD
    elif next_state[0]==ALMOST_COLLISION_STATE and next_state[1]==0:  # Check if collision: Close distance No Green
        reward += ALMOST_HIT_PENALTY
        if action == 'left':
            reward +=  LEFT_REWARD
    elif next_state[0]==COLLISION_STATE and next_state[1]==0:  # Check if collision: Close distance No Green
        reward += HIT_PENALTY
        if action == 'left':
            reward +=  LEFT_REWARD
    elif action == 'forward':
        reward += FORWARD_REWARD
    else:
        reward += 1  # Default reward

    # if falls of map, sensors are 0, then stop simulation
    if next_state[0] == 0:
        done = True
    else: done = False
    """
    return next_state, reward, done


# Training function using Q-learning
def train_q_table(rob, run_name, q_table, q_table_path,results_path, num_episodes=200, max_steps=40, alpha=0.1, gamma=0.9, epsilon=0.1):
    global TRAINING_RESULTS
    # Initialize Training Object and CSV to store results
    clear_csv(results_path)
    TRAINING_RESULTS = Training_Results.Training_Results(run_name=run_name,
                                        ir_bin_thresholds=IR_BIN_THRESHOLDS,
                                        green_bin_thresholds=GREEN_BIN_THRESHOLDS,
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
        rob.set_phone_tilt_blocking(109, 60)
        rob.move_blocking(50,50,400)

        # Build up state
        state_img = get_state_img(rob, str(FIGRURES_DIR / "state_image_test1.png"))
        state_greenness, greenness_direction = get_state_greenness(state_img)

        state = (1,state_greenness, greenness_direction)        
        done = False

        for step in range(max_steps):
            print("Episode: ", episode, "Step: ", step)
            TRAINING_RESULTS.steps['episode'] = episode
            TRAINING_RESULTS.steps['step'] = step

            # Choose an action, random by prob. epsilon, max, by prob 1-epsilon
            action, action_index = get_action(q_table, state, epsilon)
            TRAINING_RESULTS.steps['action'] = action

            # Take the action and observe the new state and reward  state:[IR, %Green, Direction Green]
            new_state, reward, done = simulate_robot_action(rob, action)
            if new_state[1]>state[1]: reward += GREEN_REWARD
            if new_state[1]<state[1] and action != "forward" : reward -= 2*GREEN_REWARD

            print(f"Moved from state {state} to {new_state} by going {action}, got new reward {reward}")
            TRAINING_RESULTS.steps['new_state'] = new_state
            TRAINING_RESULTS.steps['reward'] = reward

            # Update the Q-value in Q-table
            max_future_q = max(q_table[new_state])
            current_q = q_table[state][action_index]
            q_table[state][action_index] = current_q + alpha * (reward + gamma * max_future_q - current_q)

            # Check collision with object, or maximum steps is reached, then stop simulation
            objects_found = 0
            if step >= max_steps-1:
                objects_found = rob.nr_food_collected()
                print(f"/n I found {objects_found} food objects!!")
                rob.talk(f"Hello, I found {objects_found} food objects!!")

                done = True
                if isinstance(rob, SimulationRobobo):
                    rob.stop_simulation()
                print(f"------------------- END EPISODE {episode} --------------------")

            TRAINING_RESULTS.steps['objects_found'] = objects_found
            
            # Store result in csv.
            TRAINING_RESULTS.write_line_to_csv(results_path)

            # Transition to the new state
            state = new_state

            if done:
                break

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

    # Build up state
    state_img = get_state_img(rob, str(FIGRURES_DIR / "state_image_test1.png"))
    state_greenness, greenness_direction = get_state_greenness(state_img)

    state = (1,state_greenness, greenness_direction)        
    done = False

    while True:
        # Determine action by state from q-table
        action, action_index = get_action(q_table, state, epsilon)

        # Take the action and observe the new state
        if hardware_flag:
            new_state = play_robot_action_hardware(rob, action, thresholds=IR_BIN_THRESHOLDS_HARDWARE)

        else:
            new_state = play_robot_action(rob, action)

        print(f"Moved from state {state} to {new_state} by going {action}.")

        # Transition to the new state
        state = new_state

        # Check if robot has collided, then stop simulation
        #if rob.nr_food_collected == 5: 
         #   done = True

          #  rob.talk(f"Hello, I found {rob.nr_food_collected()} food objects!!")


#            if isinstance(rob, SimulationRobobo):
 #               rob.stop_simulation()

        if done:
            break

    print_q_table(q_table)

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



# Function to test actions
def test_robo2(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        print("Start simulation")
    
    rob.set_phone_tilt_blocking(109, 60)
    
    # rob.move_blocking(50, -10, 920) # 90 Degrees Right
    # rob.move_blocking(-10, 50, 460) # 45 Degrees Left
    # rob.move_blocking(60, 60, 2300) # 1 Grid forward
    
    rob.move_blocking(-15, 25, 200)
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "test_red_block_rgb.png"), image)
    rob.move_blocking(25, -15, 200)
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "test_red_block_rgb.png"), image)
    rob.move_blocking(25, -15, 200)
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "test_red_block_rgb.png"), image)
    rob.move_blocking(25, -15, 200)
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "test_red_block_rgb.png"), image)
    rob.move_blocking(25, -15, 200)
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "test_red_block_rgb.png"), image)
    rob.move_blocking(25, -15, 200)
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "test_red_block_rgb.png"), image)
    rob.move_blocking(25, -15, 200)
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "test_red_block_rgb.png"), image)
    rob.move_blocking(25, -15, 200)
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "test_red_block_rgb.png"), image)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()









    # Build up state

    ################## Sketch 1)

    # Getting RED Object
    # State[ Center_IR, % Red, Direction Red]       : Action ['left', 'forward', 'right'] (pivot/move to next direction bin)

    # Getting to GREEN surface
    # State[ Center_IR, % Green, Direction Green]   : Action ['left', 'forward', 'right']


    # Pro:  Could steer into Objective
    # Con:  Difficult reward function



    ################## Sketch 2)
    # Pro:  Simple reward function
    # Con:  Assumes hardware drives straight forward 

    # Searching Objective
    # State[ Center_IR, Red/Green Boolean, % Colour, Direction Colour]       : Action ['left', 'right'] (pivot to next direction bin)

    # Drive
    # State[ Center_IR, % Color]   : Action ['forward']