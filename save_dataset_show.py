import asyncio
import aiohttp
import aiofiles
import os
import time
import base64
import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

# =================================
# 配置部分
# =================================
BASE_URL = 'http://localhost:8000'
DATA_DIR = 'TUM_Dataset'
INTERVALS = {
    'control': 0.1,        # 10Hz
    'rpm': 0.1,            # 10Hz
    'imu_gps': 1.0,        # 1Hz
    'imu_accel': 0.01,     # 100Hz
    'camera': 1/20,        # 20 FPS
}

# 日志配置
logging.basicConfig(
    level=logging.WARNING,  # 为高频任务减少日志输出
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =================================
# 目录和 CSV 初始化
# =================================
def create_dirs():
    """
    创建数据存储所需的目录。
    """
    dirs = [
        'rgb/front',
        'rgb/rear',
        'imu',
        'gps',
        'control',
        'rpm',
        'videos'
    ]
    for d in dirs:
        path = os.path.join(DATA_DIR, d)
        try:
            os.makedirs(path, exist_ok=True)
            logger.debug(f"目录已创建或已存在: {path}")
        except Exception as e:
            logger.error(f"无法创建目录 {path}: {e}")

async def initialize_csv():
    """
    初始化 CSV 文件，如果不存在则创建，并添加适当的头部。
    """
    files_and_headers = {
        os.path.join(DATA_DIR, 'control', 'control.csv'): ['timestamp', 'speed', 'orientation'],
        os.path.join(DATA_DIR, 'rpm', 'rpm.csv'): ['timestamp', 'rpm1', 'rpm2', 'rpm3', 'rpm4'],
        os.path.join(DATA_DIR, 'imu', 'imu_gps.csv'): [
            'timestamp', 'latitude', 'longitude', 'gx', 'gy', 'gz', 'mx', 'my', 'mz'
        ],
        os.path.join(DATA_DIR, 'imu', 'accel_high.csv'): ['timestamp', 'ax', 'ay', 'az'],
    }
    for file, headers in files_and_headers.items():
        if not os.path.exists(file):
            try:
                async with aiofiles.open(file, 'w') as csvfile:
                    header_line = ','.join(headers) + '\n'
                    await csvfile.write(header_line)
                logger.info(f"已初始化 CSV 文件: {file}")
            except Exception as e:
                logger.error(f"无法初始化 CSV 文件 {file}: {e}")

# =================================
# 共享数据结构
# =================================
QUEUE_MAXSIZE = 1000000
control_queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
rpm_queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
imu_gps_queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
imu_accel_queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

# 定义信号量以限制并发的图像写入操作
IMAGE_WRITE_CONCURRENCY = 10  # 根据系统性能调整
image_write_semaphore = asyncio.Semaphore(IMAGE_WRITE_CONCURRENCY)

# =================================
# 辅助函数
# =================================
async def write_csv_lines(file_path, lines):
    """
    将多行写入 CSV 文件。
    """
    try:
        async with aiofiles.open(file_path, 'a') as csvfile:
            await csvfile.writelines(lines)
    except Exception as e:
        logger.error(f"无法写入 {file_path}: {e}")

# =================================
# 统一数据获取函数
# =================================
async def fetch_data(session):
    """
    以最高所需频率从服务器获取数据并进行分发。
    """
    interval = INTERVALS['imu_accel']  # 最高频率：100Hz

    # 初始化上次处理的时间
    last_times = {
        'control': 0.0,
        'rpm': 0.0,
        'imu_gps': 0.0,
    }

    while True:
        start_time = time.monotonic()
        try:
            async with session.get(f'{BASE_URL}/data', timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    timestamp = float(data.get('timestamp', time.time()))

                    current_time = time.monotonic()

                    # 每 0.1 秒处理一次控制数据（10Hz）
                    if current_time - last_times['control'] >= INTERVALS['control']:
                        last_times['control'] = current_time
                        control_data = {
                            'timestamp': timestamp,
                            'linear': data.get('speed', 0),
                            'angular': data.get('orientation', 0)
                        }
                        try:
                            control_queue.put_nowait(control_data)
                        except asyncio.QueueFull:
                            logger.warning("控制队列已满，丢弃数据。")

                    # 每 0.1 秒处理一次 RPM 数据（10Hz）
                    if current_time - last_times['rpm'] >= INTERVALS['rpm']:
                        last_times['rpm'] = current_time
                        rpms = data.get('rpms', [])
                        for rpm in rpms:
                            if isinstance(rpm, list) and len(rpm) == 5:
                                rpm1, rpm2, rpm3, rpm4, ts = rpm
                                rpm_data = {
                                    'timestamp': float(ts),
                                    'rpm1': rpm1,
                                    'rpm2': rpm2,
                                    'rpm3': rpm3,
                                    'rpm4': rpm4
                                }
                                try:
                                    rpm_queue.put_nowait(rpm_data)
                                except asyncio.QueueFull:
                                    logger.warning("RPM 队列已满，丢弃数据。")

                    # 每 1 秒处理一次 IMU（陀螺仪和磁力计）和 GPS 数据（1Hz）
                    if current_time - last_times['imu_gps'] >= INTERVALS['imu_gps']:
                        last_times['imu_gps'] = current_time
                        imu_gps_data = {
                            'timestamp': timestamp,
                            'latitude': data.get('latitude'),
                            'longitude': data.get('longitude'),
                            'gx': None,
                            'gy': None,
                            'gz': None,
                            'mx': None,
                            'my': None,
                            'mz': None,
                        }

                        # 处理 IMU 陀螺仪数据
                        gyros = data.get('gyros', [])
                        if gyros:
                            for gyro in gyros:
                                if isinstance(gyro, list) and len(gyro) == 4:
                                    gx, gy_val, gz, ts = gyro
                                    imu_gps_data.update({
                                        'gx': gx,
                                        'gy': gy_val,
                                        'gz': gz,
                                    })
                                    break  # 只取第一组数据

                        # 处理 IMU 磁力计数据
                        mags = data.get('mags', [])
                        if mags:
                            for mag in mags:
                                if isinstance(mag, list) and len(mag) == 4:
                                    mx, my, mz, ts = mag
                                    imu_gps_data.update({
                                        'mx': mx,
                                        'my': my,
                                        'mz': mz,
                                    })
                                    break  # 只取第一组数据

                        try:
                            imu_gps_queue.put_nowait(imu_gps_data)
                        except asyncio.QueueFull:
                            logger.warning("IMU/GPS 队列已满，丢弃数据。")

                    # 处理高频加速度数据（100Hz）
                    accels = data.get('accels', [])
                    for accel in accels:
                        if isinstance(accel, list) and len(accel) == 4:
                            ax, ay, az, ts = accel
                            imu_accel_data = {
                                'timestamp': float(ts),
                                'ax': ax,
                                'ay': ay,
                                'az': az
                            }
                            try:
                                imu_accel_queue.put_nowait(imu_accel_data)
                            except asyncio.QueueFull:
                                logger.warning("IMU 加速度队列已满，丢弃数据。")
                else:
                    logger.warning(f"无法获取数据。状态码: {response.status}")
        except Exception as e:
            logger.error(f"获取数据时出错: {e}")

        # 调整睡眠时间以维持期望的频率
        elapsed = time.monotonic() - start_time
        sleep_time = max(0, interval - elapsed)
        await asyncio.sleep(sleep_time)

async def fetch_camera(session, executor):
    """
    以指定的频率获取并保存前后摄像头图像。
    """
    img_dirs = {
        'front': os.path.join(DATA_DIR, 'rgb', 'front'),
        'rear': os.path.join(DATA_DIR, 'rgb', 'rear')
    }
    interval = INTERVALS['camera']
    next_time = time.monotonic()
    while True:
        start_time = time.monotonic()
        try:
            params = "?view_types=front,rear"
            async with session.get(f'{BASE_URL}/screenshot{params}', timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    timestamp = float(data.get('timestamp', time.time()))

                    for view_type in ['front', 'rear']:
                        frame_key = f"{view_type}_frame"
                        frame_base64 = data.get(frame_key, '')
                        if isinstance(frame_base64, str) and frame_base64:
                            try:
                                # 解码 base64 图像
                                img_data = base64.b64decode(frame_base64)
                                np_arr = np.frombuffer(img_data, np.uint8)
                                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                if img is not None:
                                    filename = f"{timestamp:.6f}.png"
                                    filepath = os.path.join(img_dirs[view_type], filename)
                                    # 使用信号量限制并发写入
                                    async with image_write_semaphore:
                                        loop = asyncio.get_running_loop()
                                        await loop.run_in_executor(executor, cv2.imwrite, filepath, img)
                                else:
                                    logger.warning(f"解码 {view_type} 摄像头图像失败，时间戳: {timestamp}")
                            except base64.binascii.Error as e:
                                logger.error(f"解码 {view_type} 摄像头图像的 base64 数据时出错，时间戳: {timestamp}: {e}")
                        else:
                            logger.warning(f"未收到 {view_type} 摄像头的图像数据，时间戳: {timestamp}")
                else:
                    logger.warning(f"无法获取摄像头数据。状态码: {response.status}")
        except Exception as e:
            logger.error(f"获取摄像头数据时出现意外错误: {e}")

        # 调整睡眠时间以维持期望的频率
        elapsed = time.monotonic() - start_time
        next_time += interval
        sleep_time = max(0, next_time - time.monotonic())
        await asyncio.sleep(sleep_time)

# =================================
# 数据保存函数
# =================================
async def save_control():
    """
    以指定的间隔保存控制数据，一次性写入缓存数据。
    """
    csv_file = os.path.join(DATA_DIR, 'control', 'control.csv')
    interval = 10.0  # 间隔时间（秒）
    next_time = time.monotonic()
    while True:
        try:
            data_list = []
            while not control_queue.empty():
                control_data = control_queue.get_nowait()
                data_list.append(control_data)
            if data_list:
                lines = []
                for control_data in data_list:
                    timestamp = control_data['timestamp']
                    linear = control_data['linear']
                    angular = control_data['angular']
                    if isinstance(linear, (int, float)) and isinstance(angular, (int, float)):
                        line = f"{timestamp},{linear},{angular}\n"
                        lines.append(line)
                    else:
                        logger.warning("控制数据字段的数据类型无效。")
                if lines:
                    await write_csv_lines(csv_file, lines)
        except Exception as e:
            logger.error(f"保存控制数据时出错: {e}")

        # 调整睡眠时间以维持期望的间隔
        next_time += interval
        sleep_time = max(0, next_time - time.monotonic())
        await asyncio.sleep(sleep_time)

async def save_rpm():
    """
    以指定的间隔保存 RPM 数据，一次性写入缓存数据。
    """
    rpm_file = os.path.join(DATA_DIR, 'rpm', 'rpm.csv')
    interval = 10.0  # 间隔时间（秒）
    next_time = time.monotonic()
    while True:
        try:
            rpm_lines = []
            while not rpm_queue.empty():
                rpm_data = rpm_queue.get_nowait()
                timestamp = rpm_data['timestamp']
                rpm1 = rpm_data['rpm1']
                rpm2 = rpm_data['rpm2']
                rpm3 = rpm_data['rpm3']
                rpm4 = rpm_data['rpm4']
                if all(isinstance(val, (int, float)) for val in [rpm1, rpm2, rpm3, rpm4, timestamp]):
                    line = f"{timestamp},{rpm1},{rpm2},{rpm3},{rpm4}\n"
                    rpm_lines.append(line)
                else:
                    logger.warning("RPM 数据字段的数据类型无效。")
            if rpm_lines:
                await write_csv_lines(rpm_file, rpm_lines)
        except Exception as e:
            logger.error(f"保存 RPM 数据时出错: {e}")

        # 调整睡眠时间以维持期望的间隔
        next_time += interval
        sleep_time = max(0, next_time - time.monotonic())
        await asyncio.sleep(sleep_time)

async def save_imu_gps():
    """
    以指定的间隔保存 IMU（陀螺仪和磁力计）和 GPS 数据，一次性写入缓存数据。
    """
    csv_file = os.path.join(DATA_DIR, 'imu', 'imu_gps.csv')
    interval = 10.0  # 间隔时间（秒）
    next_time = time.monotonic()
    while True:
        try:
            data_list = []
            while not imu_gps_queue.empty():
                data = imu_gps_queue.get_nowait()
                data_list.append(data)
            if data_list:
                lines = []
                for data in data_list:
                    timestamp = data['timestamp']
                    latitude = data.get('latitude')
                    longitude = data.get('longitude')
                    gx = data.get('gx')
                    gy_val = data.get('gy')
                    gz = data.get('gz')
                    mx = data.get('mx')
                    my = data.get('my')
                    mz = data.get('mz')
                    line = f"{timestamp},{latitude},{longitude},{gx},{gy_val},{gz},{mx},{my},{mz}\n"
                    lines.append(line)
                if lines:
                    await write_csv_lines(csv_file, lines)
        except Exception as e:
            logger.error(f"保存 IMU/GPS 数据时出错: {e}")

        # 调整睡眠时间以维持期望的间隔
        next_time += interval
        sleep_time = max(0, next_time - time.monotonic())
        await asyncio.sleep(sleep_time)

async def save_accel():
    """
    以指定的间隔保存高频加速度数据，一次性写入缓存数据。
    """
    accel_high_file = os.path.join(DATA_DIR, 'imu', 'accel_high.csv')
    interval = 10.0  # 间隔时间（秒）
    next_time = time.monotonic()
    while True:
        try:
            accel_lines = []
            while not imu_accel_queue.empty():
                accel_data = imu_accel_queue.get_nowait()
                timestamp = accel_data['timestamp']
                ax = accel_data['ax']
                ay = accel_data['ay']
                az = accel_data['az']
                if all(isinstance(val, (int, float)) for val in [ax, ay, az, timestamp]):
                    line = f"{timestamp},{ax},{ay},{az}\n"
                    accel_lines.append(line)
                else:
                    logger.warning("加速度数据的数据类型无效。")
            if accel_lines:
                await write_csv_lines(accel_high_file, accel_lines)
        except Exception as e:
            logger.error(f"保存加速度数据时出错: {e}")

        # 调整睡眠时间以维持期望的间隔
        next_time += interval
        sleep_time = max(0, next_time - time.monotonic())
        await asyncio.sleep(sleep_time)

# =================================
# 主函数
# =================================
async def main():
    """
    主函数，初始化目录和 CSV，并启动所有数据采集和保存的协程。
    """
    create_dirs()
    await initialize_csv()

    # 创建 ThreadPoolExecutor，用于处理阻塞操作，如 cv2.imwrite
    with ThreadPoolExecutor(max_workers=20) as executor:
        async with aiohttp.ClientSession() as session:
            # 使用 asyncio.create_task() 调度所有协程
            tasks = [
                asyncio.create_task(fetch_data(session)),
                asyncio.create_task(save_control()),
                asyncio.create_task(save_rpm()),
                asyncio.create_task(save_imu_gps()),
                asyncio.create_task(save_accel()),
                asyncio.create_task(fetch_camera(session, executor)),
            ]

            logger.info("所有数据采集任务已启动。")

            # 等待所有任务完成（它们将无限期运行）
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"主循环中发生错误: {e}")

if __name__ == "__main__":
    try:
        # 确保我们使用相同的事件循环
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("收到键盘中断信号。停止数据采集...")
    except Exception as e:
        logger.error(f"发生意外错误: {e}")

