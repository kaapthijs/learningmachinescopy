#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import initialize_q_table,print_q_table,train_q_table_phase1, play_q_table_phase1, train_q_table_phase2, play_q_table_phase2, load_q_table, test_robo
#from learning_machines import test_take_picture

from data_files import FIGRURES_DIR
from data_files import RESULT_DIR

if __name__ == "__main__":
    # You can do better argument parsing than this!
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    # SET RESULT NAMES
    RUN_NAME = "TASK3_Dest_1"
    q_table_path = str(RESULT_DIR)  + '/' + RUN_NAME + "_Q_table.pkl"
    result_path = str(RESULT_DIR) + '/'  + RUN_NAME + "_Results.csv"

    #print(q_table_path)
    # Load or initialize the Q-table
    #q_table = initialize_q_table(q_table_path=q_table_path)

    #print("Initial Q-table:")
    #print_q_table(q_table, num_entries=60)

    # Train the Q-table
    #train_q_table_object(rob, RUN_NAME, q_table, q_table_path, result_path, num_episodes=15, max_steps=20, epsilon=0.30)
    #train_q_table_destination(rob, RUN_NAME, q_table, q_table_path, result_path, num_episodes=15, max_steps=20, epsilon=0.30)

    trained_q_table = load_q_table(q_table_path="/root/results/TASK2_Training_Thijs3_Q_table.pkl")
    #print_q_table(trained_q_table, num_entries=60)

    #play_q_table_object(rob, trained_q_table, epsilon=0.05, hardware_flag=True)
    play_q_table_phase2(rob, trained_q_table, epsilon=0.05, hardware_flag=True)
    
    #test_robo(rob)