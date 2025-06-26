class ServoDriver:
    def __init__(self, servo_config):
        self.n_servos = servo_config['n_servos']
        self.origin = servo_config['origin']
        self.ceiling = servo_config['ceiling']
        self.total_positions = servo_config['total_positions']
        self.current_positions = [0] * self.n_servos

    def move_to_position(self, target_positions):
        for i in range(self.n_servos):
            if self.validate_position(target_positions[i]):
                self.current_positions[i] = target_positions[i]
                self.execute_movement(i, target_positions[i])
            else:
                raise ValueError(f"Target position {target_positions[i]} for servo {i} is out of bounds.")

    def validate_position(self, position):
        return self.origin[0] <= position <= self.ceiling[0]

    def execute_movement(self, servo_index, target_position):
        # Code to interface with the actual servo hardware
        print(f"Moving servo {servo_index} to position {target_position}")

    def get_current_positions(self):
        return self.current_positions

    def calibrate(self):
        # Code to calibrate the servo positions
        print("Calibrating servos to origin positions.")
        self.current_positions = self.origin.copy()