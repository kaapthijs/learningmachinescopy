import pickle
import numpy as np
import cv2

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

# Define actions
actions = ['forward', 'left', 'right']

def load_q_table(file_path='q_table.pkl'):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def save_q_table(q_table, file_path='q_table.pkl'):
    with open(file_path, 'wb') as f:
        pickle.dump(q_table, f)

def choose_action(state, q_table, epsilon=0.1):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(len(actions))
    else:
        return np.argmax(q_table[state])

def get_state_from_ir_values(ir_values):
    # Define the bin edges and corresponding state values
    def bin_value(value):
        if value < 6:
            return 0  # No detection
        elif value < 10:
            return 1  # Detected 
        elif value <= 15:
            return 2  # Detected close
        else:
            return 3  # Hits object
    
    # Convert each sensor value to its bin
    binned_values = tuple(bin_value(value) for value in ir_values)
    
    return binned_values

def navigate_with_q_learning(rob):
    q_table = load_q_table()
    
    while True:
        ir_values = rob.read_irs()
        state = get_state_from_ir_values(ir_values)
        action_index = choose_action(state, q_table)
        action = actions[action_index]

        if action == 'forward':
            rob.move(50, 50, 1000)
        elif action == 'left':
            rob.move(50, -50, 800)
        elif action == 'right':
            rob.move(-50, 50, 800)

        rob.sleep(1)

def simulate_robot_action(rob, action):
    if action == 'forward':
        rob.move(50, 50, 1000)
    elif action == 'left':
        rob.move(50, -50, 800)
    elif action == 'right':
        rob.move(-50, 50, 800)
    rob.sleep(1)

    ir_values = rob.read_irs()
    next_state = get_state_from_ir_values(ir_values)
    reward = -1 if any(value > 15 for value in ir_values) else 1  # Penalty for hitting object, reward otherwise
    
    return next_state, reward

def train_q_table(rob, q_table, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    for episode in range(num_episodes):
        # Reset the environment and get the initial state
        rob.reset()  # Assume we have a reset method to start from a known state
        ir_values = rob.read_irs()
        state = get_state_from_ir_values(ir_values)

        if isinstance(rob, SimulationRobobo): 
            rob.play_simulation()


        done = False
        while not done:
            # Choose an action using epsilon-greedy policy
            action_index = choose_action(state, q_table, epsilon)
            action = actions[action_index]

            # Take the action and observe the next state and reward
            next_state, reward = simulate_robot_action(rob, action)
            
            # Update the Q-table
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][action_index] += alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][action_index])
            
            state = next_state

            if reward == -1:  # Assuming the episode ends if the robot hits an object
                done = True

    # Save the trained Q-table
    save_q_table(q_table)