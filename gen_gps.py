import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import folium

# =================================
# 配置部分
# =================================
DATA_DIR = 'TUM_Dataset'
IMU_GPS_CSV = os.path.join(DATA_DIR, 'imu', 'imu_gps.csv')

# 日志配置
logging.basicConfig(
    level=logging.INFO,  # 信息级别日志
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =================================
# 轨迹生成函数
# =================================
def generate_static_trajectory(csv_file, output_image='trajectory.png'):
    """
    读取CSV文件并生成静态的轨迹图。
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 检查必要的列是否存在
        if not {'latitude', 'longitude', 'timestamp'}.issubset(df.columns):
            logger.error("CSV文件缺少必要的列（latitude, longitude, timestamp）。")
            return

        # 丢弃缺失的GPS数据
        df = df.dropna(subset=['latitude', 'longitude'])

        # 绘制轨迹
        plt.figure(figsize=(10, 8))
        plt.plot(df['longitude'], df['latitude'], marker='o', linestyle='-', markersize=2, linewidth=1)
        plt.title('车辆移动轨迹')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.grid(True)
        plt.savefig(output_image)
        plt.close()
        logger.info(f"静态轨迹图已保存为 {output_image}")
    except Exception as e:
        logger.error(f"生成静态轨迹图时出错: {e}")

def generate_interactive_map(csv_file, output_html='trajectory_map.html'):
    """
    读取CSV文件并生成交互式的轨迹地图。
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 检查必要的列是否存在
        if not {'latitude', 'longitude', 'timestamp'}.issubset(df.columns):
            logger.error("CSV文件缺少必要的列（latitude, longitude, timestamp）。")
            return

        # 丢弃缺失的GPS数据
        df = df.dropna(subset=['latitude', 'longitude'])

        # 获取轨迹的起点作为地图的中心
        start_lat = df.iloc[0]['latitude']
        start_lon = df.iloc[0]['longitude']

        # 创建Folium地图
        m = folium.Map(location=[start_lat, start_lon], zoom_start=15)

        # 添加轨迹线
        folium.PolyLine(df[['latitude', 'longitude']].values, color="blue", weight=2.5, opacity=1).add_to(m)

        # 添加起点和终点标记
        folium.Marker([start_lat, start_lon], popup='起点', icon=folium.Icon(color='green')).add_to(m)
        end_lat = df.iloc[-1]['latitude']
        end_lon = df.iloc[-1]['longitude']
        folium.Marker([end_lat, end_lon], popup='终点', icon=folium.Icon(color='red')).add_to(m)

        # 保存为HTML文件
        m.save(output_html)
        logger.info(f"交互式轨迹地图已保存为 {output_html}")
    except Exception as e:
        logger.error(f"生成交互式轨迹地图时出错: {e}")

def main():
    """
    主函数，调用轨迹生成函数。
    """
    if not os.path.exists(IMU_GPS_CSV):
        logger.error(f"找不到CSV文件: {IMU_GPS_CSV}")
        return

    # 生成静态轨迹图
    generate_static_trajectory(IMU_GPS_CSV, output_image='trajectory.png')

    # 生成交互式轨迹地图
    generate_interactive_map(IMU_GPS_CSV, output_html='trajectory_map.html')

if __name__ == "__main__":
    main()

