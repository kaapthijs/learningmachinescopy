import cv2

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


def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)


def test_move_and_wheel_reset(rob: IRobobo):
    rob.move_blocking(100, 100, 1000)
    print("before reset: ", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(1)
    print("after reset: ", rob.read_wheels())


def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())


def test_phone_movement(rob: IRobobo):
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())


def test_sim(rob: SimulationRobobo):
    print(rob.get_sim_time())
    print(rob.is_running())
    rob.stop_simulation()
    print(rob.get_sim_time())
    print(rob.is_running())
    rob.play_simulation()
    print(rob.get_sim_time())
    print(rob.get_position())


def test_hardware(rob: HardwareRobobo):
    print("Phone battery level: ", rob.read_phone_battery())
    print("Robot battery level: ", rob.read_robot_battery())

def test_phone_moving(rob: IRobobo):
    # print current position
    print(f"Camera starts in pan: {rob.read_phone_pan()}")
    print(f"Camera starts in tilt: {rob.read_phone_tilt()}")

    # test moving pan and tilt
    ## pan(pan:[11-343], speed:[0-100])
    ## tilt(tilt:[26-109], speed:[0-100])
    
    rob.set_phone_pan_blocking(11, 20)
    print("Phone pan after move to 11: ", rob.read_phone_pan())
    rob.set_phone_pan_blocking(340, 20)
    print("Phone pan after move to 340: ", rob.read_phone_pan())
    
    rob.set_phone_tilt_blocking(100, 20)
    print("Phone tilt after move to 100: ", rob.read_phone_tilt())
    rob.set_phone_tilt_blocking(28, 20)
    print("Phone tilt after move to 28: ", rob.read_phone_tilt())

def test_take_picture(rob: IRobobo):
    # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:

    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "test_green_block_rgb.png"), image)

    print("Image shape:", image.shape)
    print(image)

    # Print the pixel values at the four corners
    print("Top-left corner pixel value:", image[0, 0])
    print("Top-right corner pixel value:", image[0, -1])
    print("Bottom-left corner pixel value:", image[-1, 0])
    print("Bottom-right corner pixel value:", image[-1, -1])
    
def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    #test_emotions(rob)
    
    # test taking picture and store
    rob.set_phone_tilt_blocking(109, 20)
    rob.set_phone_pan_blocking(11, 20)
    test_take_picture(rob)

    #test_sensors(rob)
    #test_phone_movement(rob)
    
    #test_move_and_wheel_reset(rob)
    #if isinstance(rob, SimulationRobobo):
    #    test_sim(rob)

    #if isinstance(rob, HardwareRobobo):
    #    test_hardware(rob)

    

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

def move_robot(rob: IRobobo):
    """Moves the robot straight until an object is detected, then turns right."""
    if isinstance(rob, SimulationRobobo): 
        rob.play_simulation()

    ir_values = rob.read_irs()

    # IR sensors detect an object
    if all(value < 20 for value in ir_values[2:6]):  # threshold value 
        # Stop moving forward
        rob.move(0, 0, 0)
        rob.sleep(0.5)

        # Turn right
        rob.move(50, -50, 1000) 
        #rob.block()

    else:
        # Move forward
        rob.move(50, 50, 1000)
        #rob.block()
    
    rob.sleep(0.1)

        #if isinstance(rob, SimulationRobobo):
            #rob.stop_simulation()

def avoid_object(rob: IRobobo):
    """
    This function directs the robot to move straight until it detects an object nearby.
    Upon detection, the robot will change its emotion to indicate awareness or caution,
    then turn right to avoid the object.
    """
    if isinstance(rob, SimulationRobobo): 
        rob.play_simulation()

    while True:
        ir_values = rob.read_irs()
        print("ir value: ", ir_values[4])

        # Check if any of the front sensors detect an object within a close range
        #if any(value > 15 for value in ir_values[2:4]):  # threshold value as needed
        if ir_values[4] > 15 and ir_values[4] < 1500:
            rob.set_emotion(Emotion.SURPRISED)  # change the robot's emotion
            rob.talk("Oh object detected!")  # have the robot acknowledge the detection
            
            # Stop moving forward
            rob.move(0, 0, 0)
            rob.sleep(0.5)
            
            # Turn right
            rob.move(-50, 50, 800)  # speed values
            rob.sleep(1)
            #break  # Exit the loop if you only want to avoid one obstacle, or continue if scanning for more

        else:
            # Continue moving forward
            rob.move(50, 50, 1000)

    rob.sleep(1)  

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()