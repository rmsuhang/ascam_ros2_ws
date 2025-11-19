from math import frexp
from torch import imag
from yolov5 import YOLOv5
import torch

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import (
    QoSProfile,               # QoS配置类
    QoSReliabilityPolicy,     # 可靠性策略
    QoSDurabilityPolicy,      # 持久性策略
    QoSHistoryPolicy,         # 历史策略
    qos_profile_sensor_data   # 预设的传感器QoS配置
)
import cv2
import os

class CarriageBaffleNode(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        #加载模型
        package_share_dir = get_package_share_directory("python_algorithms")
        model_path = os.path.join(package_share_dir, "models", "front_baffle.pt")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.yolov5 = YOLOv5(model_path=model_path, device = device)
        self.get_logger().info(f"Using device: {device}")

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        #创建发布者
        self.yolo_result_pub = self.create_publisher(
            Detection2DArray, "yolo_result", 10)
        self.result_msg = Detection2DArray()
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)

        self.image_sub = self.create_subscription(
            Image, "/cloud_image", self.image_callback, qos_profile=qos_profile)


        self.bridge = CvBridge()

    def image_callback(self, msg) -> None:

        self.get_logger().info("接受到点云数据，触发回调")
        # ROS图像转OpenCV格式
        # YOLOv5检测
        mono8_img = self.bridge.imgmsg_to_cv2(msg,  desired_encoding='mono8')
        image = cv2.cvtColor(mono8_img, cv2.COLOR_GRAY2BGR)
        detect_result = self.yolov5.predict(image)
        self.get_logger().info(str(detect_result))

        #清空历史缓存列表
        #设置坐标系 设置时间戳
        self.result_msg.detections.clear()
        self.result_msg.header.frame_id = "rslidar"
        self.result_msg.header.stamp =  msg.header.stamp

         # 解析检测结果（边界框、置信度、类别）
        predictions = detect_result.pred[0]
        boxes = predictions[:, :4]  # [x1, y1, x2, y2]
        scores = predictions[:, 4]  # 置信度
        categories = predictions[:, 5]  # 类别ID

        for index in range(len(categories)):
            name = detect_result.names[int(categories[index])]
            detection2d = Detection2D()
            detection2d.id = name
            x1, y1, x2, y2 = boxes[index]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            center_x = (x1+x2)/2.0
            center_y = (y1+y2)/2.0

            detection2d.bbox.center.position.x = center_x
            detection2d.bbox.center.position.y = center_y
            detection2d.bbox.size_x = float(x2 - x1)
            detection2d.bbox.size_y = float(y2 - y1)

            obj_pose = ObjectHypothesisWithPose()
            obj_pose.hypothesis.class_id = name
            obj_pose.hypothesis.score = float(scores[index])
            detection2d.results.append(obj_pose)
            self.result_msg.detections.append(detection2d)


            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{name}({center_x:.0f},{center_y:.0f})", (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        #发布检测图像
        result_img_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        result_img_msg.header = msg.header
        self.result_img_pub.publish(result_img_msg)

        if len(categories) > 0:
            self.yolo_result_pub.publish(self.result_msg)

def main() -> None:
    rclpy.init()
    rclpy.spin(CarriageBaffleNode("baffle_node"))
    rclpy.shutdown()

if __name__ == "__main__":
    main()



