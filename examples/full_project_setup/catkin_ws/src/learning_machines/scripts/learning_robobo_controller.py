#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import train_q_table, initialize_q_table, print_q_table, load_q_table, play_q_table

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
    RUN_NAME = "Test_Run_1"
    q_table_path = RUN_NAME + "_Q_table.pkl"
    result_path = RUN_NAME + "_Results.csv"

    # Load or initialize the Q-table
    q_table = initialize_q_table(q_table_path=q_table_path)

    # Print the initial Q-table
    print("Initial Q-table:")
    print_q_table(q_table)

    # Train the Q-table
    train_q_table(rob, RUN_NAME, q_table, q_table_path, result_path, num_episodes=2)
    
    # Print the trained Q-table
    trained_q_table = load_q_table(q_table_path=q_table_path)
    print("Trained Q-table:")
    print_q_table(trained_q_table, num_entries = 81)

    play_q_table(rob, trained_q_table)
