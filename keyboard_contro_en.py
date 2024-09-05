from pynput import keyboard
import requests
import threading

# Define the URL for controlling the car
control_url = 'http://localhost:8000/control'

# Define the function for sending control commands
def send_command(linear, angular):
    data = {
        "command": {
            "linear": linear,
            "angular": angular
        }
    }
    response = requests.post(control_url, json=data)
    if response.status_code == 200:
        print(f'Successfully sent command: linear={linear}, angular={angular}')
    else:
        print(f'Failed to send command: {response.status_code}')

# Global variables to store the state of each direction
linear_state = 0
angular_state = 0
emergency_stop = False  # Indicates the state of emergency stop
command_thread = None  # Manages the command sending thread

# Thread control flags
thread_flags = {
    'w': threading.Event(),
    's': threading.Event(),
    'a': threading.Event(),
    'd': threading.Event()
}

# Thread function for continuously sending commands at 0.1 second intervals
def command_sender():
    global linear_state, angular_state, emergency_stop
    while not emergency_stop:
        send_command(linear_state, angular_state)
        threading.Event().wait(0.1)  # Send a command every 0.1 seconds

# Start the command sending thread
def start_command_sender():
    global command_thread
    if command_thread is None or not command_thread.is_alive():
        command_thread = threading.Thread(target=command_sender, daemon=True)
        command_thread.start()

# Handle keyboard press events
def on_press(key):
    global linear_state, angular_state, emergency_stop
    try:
        if emergency_stop and key.char != 'i':
            return  # Ignore all keys except 'I' during emergency stop mode

        if key.char == 'w':
            linear_state = 1  # Move forward
            start_command_sender()

        elif key.char == 's':
            linear_state = -1  # Move backward
            start_command_sender()

        elif key.char == 'a':
            angular_state = 1  # Turn left
            start_command_sender()

        elif key.char == 'd':
            angular_state = -1  # Turn right
            start_command_sender()

        elif key.char == 'o':
            # Emergency stop
            emergency_stop = True
            linear_state = 0
            angular_state = 0
            send_command(0, 0)  # Immediately send stop command
            print("Emergency stop activated!")

        elif key.char == 'i':
            # Exit emergency stop mode and resume normal operation
            emergency_stop = False
            print("Exited emergency stop mode, resuming normal operation.")
            start_command_sender()  # Resume command sending

    except AttributeError:
        pass

# Handle keyboard release events
def on_release(key):
    global linear_state, angular_state, emergency_stop
    try:
        if emergency_stop:
            return  # Ignore other key release events if emergency stop is triggered

        if key.char == 'w':
            linear_state = 0  # Stop moving forward

        elif key.char == 's':
            linear_state = 0  # Stop moving backward

        elif key.char == 'a':
            angular_state = 0  # Stop turning left

        elif key.char == 'd':
            angular_state = 0  # Stop turning right

        # Immediately send stop command
        send_command(linear_state, angular_state)
    except AttributeError:
        pass

    if key == keyboard.Key.esc:
        # Stop listening
        print("Exiting control...")
        return False

# Listen to keyboard events
def control_car():
    print("Use 'WASD' keys to control the car. Hold to move, release to stop that direction.")
    print("Press 'O' for emergency stop. Press 'I' to exit emergency stop mode. Press 'Esc' to exit.")
    # Start listening to keyboard events
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == '__main__':
    control_car()

