from pynput import keyboard
import requests

# 定义小车控制的URL
control_url = 'http://localhost:8000/control'

# 定义控制命令的函数
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

# 处理键盘按下事件
def on_press(key):
    try:
        if key == keyboard.Key.up:
            send_command(1, 0)  # 前进
        elif key == keyboard.Key.down:
            send_command(-1, 0)  # 后退
        elif key == keyboard.Key.left:
            send_command(0, 1)  # 左转
        elif key == keyboard.Key.right:
            send_command(0, -1)  # 右转
    except AttributeError:
        pass

# 处理键盘松开事件
def on_release(key):
    if key == keyboard.Key.esc:
        # 停止监听
        print("Exiting control...")
        return False

# 监听键盘事件
def control_car():
    print("Use arrow keys to control the car. Press 'Esc' to exit.")
    # 开始监听键盘事件
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == '__main__':
    control_car()

