import os

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources')
RESOURCES_LIVER_REALSENSE_DIR = os.path.join(RESOURCES_DIR, 'liverRealSense')
SCALE = 1000


def nothing(x):
    pass


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

    # 創建視窗
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('alpha', 'RealSense', 1, 100, nothing)
    cv2.setTrackbarPos('alpha', 'RealSense', 100)

    circle_finder_map_img = None
    circle_finder_img = None

    # 串流迴圈
    while True:
        # 等待一對匹配的幀: (深度幀與顏色幀)
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # 跳過沒有幀
        if not depth_frame or not color_frame:
            continue

        # 將圖片轉換為 numpy 數組
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # BGR to RGB
        color_image = color_image[:, :, ::-1]

        # 再深度圖像上應用色彩映射 (圖像必須先轉換為 8 位元像素)
        scale_alpha = cv2.getTrackbarPos('alpha', 'RealSense')
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=(scale_alpha / SCALE)),
            cv2.COLORMAP_OCEAN
        )
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # 圖片轉為單通道
        circle_image = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
        # 霍夫找圓形
        circles = cv2.HoughCircles(circle_image, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=60, param2=30, minRadius=60, maxRadius=120)

        # 如果沒有尋找到圓形的圖片，放置預設圖片
        if circle_finder_img is None:
            circle_finder_img = color_image
            circle_finder_map_img = depth_colormap

        # 如果深度和顏色分辨率不同，則調整彩色圖像的大小以匹配顯示的深度圖片
        resized_color_image = color_image
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(
                color_image,
                dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                interpolation=cv2.INTER_AREA
            )
            circle_finder_img = cv2.resize(
                circle_finder_img,
                dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                interpolation=cv2.INTER_AREA
            )

            # 如果找到圓形
            if circles is not None:
                circles = np.uint16(np.around(circles))
                circle_image_map = np.array(depth_colormap)
                circle_image = np.array(resized_color_image)

                # 繪製圓形到圖片上
                for c in circles[0, :]:
                    cv2.circle(circle_image_map, (c[0], c[1]), c[2], (0, 255, 0), 2)
                    cv2.circle(circle_image_map, (c[0], c[1]), 2, (0, 0, 255), 3)
                    cv2.circle(circle_image, (c[0], c[1]), c[2], (0, 255, 0), 2)
                    cv2.circle(circle_image, (c[0], c[1]), 2, (0, 0, 255), 3)

                circle_finder_img = circle_image
                circle_finder_map_img = circle_image_map

        # 堆疊圖片為四格
        images_0 = np.hstack((resized_color_image, depth_colormap))
        images_1 = np.hstack((circle_finder_img, circle_finder_map_img))
        images = np.vstack((images_0, images_1))

        # 顯示圖像
        cv2.imshow('RealSense', images)

        # 離開迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 停止串流
    pipeline.stop()


if __name__ == '__main__':
    bag_file = os.path.join(RESOURCES_LIVER_REALSENSE_DIR, '20210924_095222.bag')
    read(bag_file)
