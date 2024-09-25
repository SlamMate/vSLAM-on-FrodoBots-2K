from pynput import keyboard
import requests
import threading
import os

# 定义小车控制的URL
control_url = 'http://localhost:8000/control'

# 定义记录日志的文件路径
log_file_path = 'control_log.csv'

# 创建日志文件（如果不存在）
if not os.path.exists(log_file_path):
    with open(log_file_path, 'w') as log_file:
        log_file.write("# timestamp linear angular\n")

# Function to get current data from the bot
def get_current_data():
    try:
        response = requests.get('http://localhost:8000/data')
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# Function to log the control command in TUM format (only timestamp, linear, and angular)
def log_command_tum_format(linear, angular):
    current_data = get_current_data()
    if current_data and "timestamp" in current_data:
        timestamp = current_data["timestamp"]  # 获取时间戳

        # TUM 格式记录，仅包括时间戳、线性和角运动
        log_entry = f"{timestamp} {linear} {angular}\n"

        # 将日志写入文件
        with open(log_file_path, 'a') as log_file:
            log_file.write(log_entry)
        
        print(f"Logged data in TUM format: {log_entry.strip()}")
    else:
        print("Failed to log command: timestamp not found in car data.")

# 定义控制命令的函数
def send_command(linear, angular):
    data = {
        "command": {
            "linear": linear,
            "angular": angular,
            "lamp": 0
        }
    }
    try:
        response = requests.post(control_url, json=data)
        if response.status_code == 200:
            print(f'Successfully sent command: linear={linear}, angular={angular}, lamp=0')
            log_command_tum_format(linear, angular)
        else:
            print(f'Failed to send command: {response.status_code}')
    except requests.exceptions.RequestException as e:
        print(f"Error sending command: {e}")

# 全局变量用于存储每个方向的状态
linear_state = 0
angular_state = 0
emergency_stop = False  # 用于表示紧急停止的状态
command_thread = None  # 用于管理命令发送线程

# 线程控制标志
thread_flags = {
    'w': threading.Event(),
    's': threading.Event(),
    'a': threading.Event(),
    'd': threading.Event()
}

# 线程函数，用于以0.1秒的间隔持续发送命令
def command_sender():
    global linear_state, angular_state, emergency_stop
    while not emergency_stop:
        send_command(linear_state, angular_state)
        threading.Event().wait(0.1)  # 每0.1秒发送一次命令

# 启动命令发送线程
def start_command_sender():
    global command_thread
    if command_thread is None or not command_thread.is_alive():
        command_thread = threading.Thread(target=command_sender, daemon=True)
        command_thread.start()

# 处理键盘按下事件
def on_press(key):
    global linear_state, angular_state, emergency_stop
    try:
        if emergency_stop and hasattr(key, 'char') and key.char != 'i':
            return  # 如果在紧急停止状态下，忽略除 'I' 键之外的所有按键

        if hasattr(key, 'char'):
            if key.char == 'w':
                linear_state = 1  # 前进
                start_command_sender()

            elif key.char == 's':
                linear_state = -1  # 后退
                start_command_sender()

            elif key.char == 'a':
                angular_state = 0.5  # 左转
                start_command_sender()

            elif key.char == 'd':
                angular_state = -0.5  # 右转
                start_command_sender()

            elif key.char == 'o':
                # 紧急停止
                emergency_stop = True
                linear_state = 0
                angular_state = 0
                send_command(0, 0)  # 立即发送停止命令
                print("Emergency stop activated!")

            elif key.char == 'i':
                # 退出紧急停止模式，恢复正常操作
                emergency_stop = False
                print("Exited emergency stop mode, resuming normal operation.")
                start_command_sender()  # 恢复命令发送

    except AttributeError:
        pass

# 处理键盘松开事件
def on_release(key):
    global linear_state, angular_state, emergency_stop
    try:
        if emergency_stop:
            return  # 如果紧急停止已经触发，忽略其他按键松开事件

        if hasattr(key, 'char'):
            if key.char == 'w':
                linear_state = 0  # 停止前进

            elif key.char == 's':
                linear_state = 0  # 停止后退

            elif key.char == 'a':
                angular_state = 0  # 停止左转

            elif key.char == 'd':
                angular_state = 0  # 停止右转

            # 立即发送停止命令
            send_command(linear_state, angular_state)
    except AttributeError:
        pass

    if key == keyboard.Key.esc:
        # 停止监听
        print("Exiting control...")
        return False

# 监听键盘事件
def control_car():
    print("Use 'WASD' keys to control the car. Hold to move, release to stop that direction.")
    print("Press 'O' for emergency stop. Press 'I' to exit emergency stop mode. Press 'Esc' to exit.")
    # 开始监听键盘事件
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == '__main__':
    control_car()

