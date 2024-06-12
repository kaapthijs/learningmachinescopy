#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions, move_robot, navigate_with_q_learning, train_q_table, initialize_q_table, print_q_table

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

    #move_robot(rob)
    #run_all_actions(rob)
    #avoid_object(rob)

    
    # Load or initialize the Q-table
    q_table = initialize_q_table()

    # Print the initial Q-table
    print("Initial Q-table:")
    print_q_table(q_table)

    # Train the Q-table
    train_q_table(rob, q_table)

    # Print the trained Q-table
    print("Trained Q-table:")
    print_q_table(q_table)