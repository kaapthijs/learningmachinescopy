import csv

class Training_Results:
    def __init__(self, run_name, color, color_bin_thresholds,num_episodes,max_steps,alpha,gamma,epsilon):
        # Training Results name
        self.run_name = run_name

        # Robot constants
        self.color = color
        self.color_bin_thresholds = color_bin_thresholds

        # Training_Results constants
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Episode values, change for every episode in training loop
        self.steps = {'episode': None,
                      'step' : None,
                      'color_direction': None,
                      'center_color_per': None,
                      'center_color_bin': None,
                      'object_view_per': None,
                      'state': None,
                      'action': None,
                      'new_state': None,
                      'reward': None}

    # function that creates a list of all variable names except 'steps'
    def get_header(self):
        constant_names = list(vars(self).keys())
        constant_names.remove('steps')
        step_variable_names = list(self.steps.keys())

        return constant_names + step_variable_names

    # Function that initializes a csv with header only
    def create_csv_with_header(self, file_dir):
        with open(file_dir, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')

            # write header
            header = self.get_header()
            writer.writerow(header)
    
           
    # Write training data line to existing CSV
    def write_line_to_csv(self, file_dir):
        with open(file_dir, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            
            # Get values of instance variables except 'steps'
            constant_values = [getattr(self, var) for var in vars(self) if var != 'steps']
            
            # Get values from the steps dictionary
            step_values = list(self.steps.values())
            
            # Combine both lists
            line = constant_values + step_values
            
            # Write the line to the CSV file
            writer.writerow(line)