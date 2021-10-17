import math
import os

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources')
RESOURCES_LIVER_REALSENSE_DIR = os.path.join(RESOURCES_DIR, 'liverRealSense')
SCALE = 1000
DEFAULT_SCALE = 100
# Width, Height
MASK_SIZE = (600, 600, 3)


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

    # # 創建視窗
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    # cv2.createTrackbar('alpha', 'RealSense', 1, 100, nothing)
    # cv2.setTrackbarPos('alpha', 'RealSense', DEFAULT_SCALE)

    # circle_finder_map_img = None
    circle_finder_img = None
    prev_circle = None

    # 串流迴圈
    while True:
        # 等待一對匹配的幀: (深度幀與顏色幀)
        frames = pipeline.poll_for_frames()
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

        # colorize_frame = colorizer.colorize(depth_frame)
        # colorize_image = np.asanyarray(ddf.get_data())
        # cv2.imshow("ddf", colorize_image)

        # 再深度圖像上應用色彩映射 (圖像必須先轉換為 8 位元像素)
        # scale_alpha = cv2.getTrackbarPos('alpha', 'RealSense')
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=(DEFAULT_SCALE / SCALE)),
            cv2.COLORMAP_OCEAN
        )
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # 圖片轉為單通道
        circle_image = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
        _, gray_image = cv2.threshold(circle_image, 127, 255, cv2.THRESH_TOZERO)
        # 霍夫找圓形
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=70, param2=30, minRadius=65, maxRadius=120)

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

            mask = np.zeros(MASK_SIZE, dtype=resized_color_image.dtype)

            # # 如果找到圓形
            if circles is not None:
                circles = np.uint16(np.around(circles))
                circle_image_map = np.array(depth_colormap)
                circle_image = np.array(resized_color_image)

                # 繪製圓形到圖片上
                for c in circles[0, :]:
                    if prev_circle is None:
                        prev_circle = (c[0], c[1], c[2])

                    distance = math.sqrt(
                        math.pow(
                            abs(max(prev_circle[0], c[0]) - min(prev_circle[0], c[0])), 2) +
                        math.pow(
                            abs(max(prev_circle[1], c[1]) - min(prev_circle[1], c[1])), 2))

                    cir = c
                    if distance > 10:
                        cir = prev_circle

                    cv2.circle(circle_image_map, (cir[0], cir[1]), cir[2], (0, 255, 0), 2)
                    cv2.circle(circle_image_map, (cir[0], cir[1]), 2, (0, 0, 255), 3)
                    cv2.circle(circle_image, (cir[0], cir[1]), cir[2], (0, 255, 0), 2)
                    cv2.circle(circle_image, (cir[0], cir[1]), 2, (0, 0, 255), 3)
                    cv2.putText(circle_image_map, str(cir[2]), (cir[0], cir[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 0), 1, cv2.LINE_AA)

                    prev_circle = (cir[0], cir[1], cir[2])

                circle_finder_img = circle_image
                circle_finder_map_img = circle_image_map

            if prev_circle is not None:
                center = (prev_circle[1], prev_circle[0])
                anchor_left_top = (int(center[0] - (MASK_SIZE[0] / 2)), int(center[1] - (MASK_SIZE[1] / 2)))
                image_size = resized_color_image.shape
                for y in range(MASK_SIZE[0]):
                    for x in range(MASK_SIZE[1]):
                        py = anchor_left_top[0] + y
                        px = anchor_left_top[1] + x

                        if px < 0 or py < 0 or py >= image_size[0] or px >= image_size[1]:
                            continue

                        mask[y][x] = resized_color_image[py][px]

                cv2.imshow("crop", mask)

        # #
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
    bag_file = os.path.join(RESOURCES_LIVER_REALSENSE_DIR, '20210924_101338.bag')
    read(bag_file)
