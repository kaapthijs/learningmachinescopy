#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import initialize_q_table,print_q_table,train_q_table, play_q_table, load_q_table, test_robo
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
    RUN_NAME = "TASK3_Test_2"
    q_table_path = str(RESULT_DIR)  + '/' + RUN_NAME + "_Q_table.pkl"
    result_path = str(RESULT_DIR) + '/'  + RUN_NAME + "_Results.csv"

    # Load or initialize the Q-table
    q_table = initialize_q_table(q_table_path=q_table_path)

    print("Initial Q-table:")
    print_q_table(q_table, num_entries=60)

    # Train the Q-table for RED
    train_q_table(rob,'red', RUN_NAME, q_table, q_table_path, result_path, num_episodes=5, max_steps=10, epsilon=0.30)

    # Train the Q-table for GREEN
    #train_q_table(rob,'green', RUN_NAME, q_table, q_table_path, result_path, num_episodes=5, max_steps=40, epsilon=0.20)


    # LOAD q_table for running  
    #trained_q_table = load_q_table(q_table_path)
    #print_q_table(trained_q_table, num_entries=60)


    #play_q_table(rob, trained_q_table, epsilon=0.05, hardware_flag=False)