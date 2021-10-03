import os

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources')
RESOURCES_LIVER_REALSENSE_DIR = os.path.join(RESOURCES_DIR, 'liverRealSense')


def read(bag_filename):
    # 建立一個 context 物件存放所有 RealSense 的處裡函示
    # noinspection PyArgumentList
    pipeline = rs.pipeline()

    # 創建串流物件
    config = rs.config()

    # 設定配置將使用文件中紀錄的設備，通過串流播放使用
    rs.config.enable_device_from_file(config, bag_filename)

    # 宣告特定設備進行影像串流 (定義的相機, 寬, 高, realsense的型態, 幀率)
    # 根據 bag 文件分辨率改變這個參數
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

    # 開啟影像串流
    pipeline.start(config)

    # 創建著色氣物件
    _ = rs.colorizer()

    # 串流迴圈
    while True:
        # 等待一對匹配的幀: (深度幀與顏色幀)
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 將圖片轉換為 numpy 數組
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # BGR to RGB
        color_image = color_image[:, :, ::-1]

        # 再深度圖像上應用色彩映射 (圖像必須先轉換為 8 位元像素)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=.03),
            cv2.COLORMAP_JET
        )
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # 如果深度和顏色分辨率不同，則調整彩色圖像的大小以匹配顯示的深度圖片
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(
                color_image,
                dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                interpolation=cv2.INTER_AREA
            )
            images = np.hstack((resized_color_image, depth_colormap))

        else:
            images = np.hstack((color_image, depth_colormap))

        # 顯示圖像
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        # 離開迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    bag_file = os.path.join(RESOURCES_LIVER_REALSENSE_DIR, '20210924_095222.bag')
    read(bag_file)
