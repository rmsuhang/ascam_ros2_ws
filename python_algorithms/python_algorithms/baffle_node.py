import torch
import cv2
import os
from math import frexp
from torch import imag
from yolov5 import YOLOv5

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CarriageBaffleNode(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        self.image_sub_ = self.create_subscription(Image, '/image_data',
                                                   self.cloud_image_callback,
                                                   10)
        self.get_logger().info('subscribe /image_data.')

        self.res_pub_ = self.create_publisher(Detection2DArray,
                                              '/detection_result', 10)
        self.get_logger().info('publisher /detection_result.')

        pkg_share_dire = get_package_share_directory('python_algorithms')
        model_path = os.path.join(pkg_share_dire, 'models', 'front_baffle.pt')
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.detection_model_ = YOLOv5(model_path=model_path, device=device)
        self.get_logger().info('{} using device {}.'.format(node_name, device))
        self.get_logger().info('{} loaded models.'.format(node_name))

        self.bridge_ = CvBridge()
        self.res_msg_ = Detection2DArray()

    def cloud_image_callback(self, msg):
        mono8_img = self.bridge_.imgmsg_to_cv2(msg, desired_encoding='mono8')
        img = cv2.cvtColor(mono8_img, cv2.COLOR_GRAY2BGR)
        detect_res = self.detection_model_.predict(img)

        self.res_msg_.detections.clear()
        self.res_msg_.header.frame_id = 'rslidar'
        self.res_msg_.header.stamp = msg.header.stamp

        predictions = detect_res.pred[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        categories = predictions[:, 5]
        for index in range(len(categories)):
            x1, y1, x2, y2 = boxes[index]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            name = detect_res.names[int(categories[index])]
            obj_pose = ObjectHypothesisWithPose()
            obj_pose.hypothesis.class_id = name
            obj_pose.hypothesis.score = float(scores[index])
            detection2d = Detection2D()
            detection2d.id = name
            detection2d.bbox.center.position.x = center_x
            detection2d.bbox.center.position.y = center_y
            detection2d.bbox.size_x = float(x2 - x1)
            detection2d.bbox.size_y = float(y2 - y1)
            detection2d.results.append(obj_pose)
            self.res_msg_.detections.append(detection2d)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{name}({center_x:.0f},{center_y:.0f})', (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if len(categories) > 0:
            self.res_pub_.publish(self.res_msg_)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(CarriageBaffleNode('carriage_baffle_node'))
    rclpy.shutdown()
