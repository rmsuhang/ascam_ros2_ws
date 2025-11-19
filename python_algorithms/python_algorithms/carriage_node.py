# import os
# import torch
# import math
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# import open3d as o3d

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import PointCloud2
# from sensor_msgs_py import point_cloud2
# from ament_index_python.packages import get_package_share_directory
# from std_msgs.msg import Float32MultiArray, Int32
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy

# from python_algorithms.pointnet_part_seg_msg import PartSegMsgGetModel as pointnet2
# from python_algorithms.carriage_node_utils import PartNormalDataset


# #可以在这个地方输出车辆的长宽高对应的拉筋位置，建立映射，新建话题
# class CarriageNode(Node):
#     def __init__(self, node_name):
#         super().__init__(node_name)

#         qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)

#         #订阅融合后的点云
#         self.image_sub_ = self.create_subscription(PointCloud2, '/cloud_registration',
#                                                    self.cloud_registration_callback,
#                                                    qos)
#         self.get_logger().info('subscribe /cloud_registration.')

#         self.carriage_pub_ = self.create_publisher(Float32MultiArray,
#                                                    '/carriage_seg_res', 10)
#         self.carriage_type_pub_ = self.create_publisher(Int32, '/carriage_type', 10)
#         self.get_logger().info('publisher /carriage_seg_res and /carriage_type.')

#         self.part_model_ = pointnet2(num_classes=2, normal_channel=False).eval()
#         share_pkg_dir = get_package_share_directory('python_algorithms')
#         part_model_path = os.path.join(share_pkg_dir, 'models', 'part_seg.pth')
#         device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         weight_dict = torch.load(part_model_path, map_location=device, weights_only=False)
#         self.part_model_.load_state_dict(weight_dict['model_state_dict'])
#         self.get_logger().info('model part_seg loaded.')

#         self.lable_ = np.array([0]).astype(np.int32)


#         # 车型映射表: [长, 宽, 高] 对应车型号
#         self.carriage_type_mapping = {
#             1: [4.2, 1.8, 1.6],   # 车型1的标准尺寸
#             2: [5.0, 2.0, 1.8],   # 车型2的标准尺寸
#             3: [6.5, 2.3, 2.2],   # 车型3的标准尺寸
#             4: [8.0, 2.5, 2.5],    # 车型4的标准尺寸
#             5: [13.5, 3.5,1.5]    # 车型5的标准尺寸
#         }
        
#         # 使用KNN分类器来匹配最接近的车型
#         self.type_classifier = self._init_type_classifier()

#     def _init_type_classifier(self):
#         """初始化车型分类器"""
#         # 准备训练数据
#         features = []
#         labels = []
        
#         for carriage_type, dimensions in self.carriage_type_mapping.items():
#             features.append(dimensions)
#             labels.append(carriage_type)
            
#             # 添加一些噪声数据以增强鲁棒性
#             for _ in range(5):
#                 noise = np.random.normal(0, 0.1, 3)  # 添加10%的噪声
#                 noisy_features = np.array(dimensions) + noise
#                 features.append(noisy_features.tolist())
#                 labels.append(carriage_type)
        
#         # 创建并训练KNN分类器
#         knn = KNeighborsClassifier(n_neighbors=1)
#         knn.fit(features, labels)
        
#         self.get_logger().info('Carriage type classifier initialized.')
#         return knn
    

#     def cloud_registration_callback(self, msg: PointCloud2):
#         xyz_arr = point_cloud2.read_points_numpy(msg,
#                                                  field_names=('x', 'y', 'z'),
#                                                  skip_nans=True)
#         pc = o3d.geometry.PointCloud()
#         pc.points = o3d.utility.Vector3dVector(xyz_arr)
#         pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
#         test_data_set = PartNormalDataset(pc, 60000, False)
#         data_loader = torch.utils.data.DataLoader(test_data_set, batch_size=1,
#                                                   shuffle=False, num_workers=0,
#                                                   drop_last=True)
#         for batch_id, (points, label, centroid, m) in enumerate(data_loader):
#             points = points.transpose(2, 1)
#             xyz_feature_point = points[:, :6, :]

#             seg_pred, _ = self.part_model_(points, self.to_categorical(label, 1))
#             seg_pred = seg_pred.cpu().data.numpy()
#             seg_pred = np.argmax(seg_pred, axis=-1)
#             seg_pred = np.concatenate([np.asarray(xyz_feature_point), seg_pred[:, None, :]],
#                                       axis=1).transpose((0, 2, 1)).squeeze(0)
#             normalized_coords = seg_pred[:, :3]
#             attributes = seg_pred[:, 3:]
#             original_coords = m * normalized_coords + centroid
#             denormalized_array = np.hstack((original_coords, attributes))

#             de_res = self.compare_result(xyz_arr, denormalized_array)
#             self.pub_carriage_info(de_res)

#     def to_categorical(self, y, num_classes):
#         new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
#         if y.is_cuda:
#             return new_y.cuda()
#         return new_y


#     def pub_carriage_info(self, pred_data):
#         selected = pred_data[pred_data[:, 3] == 1, :3]
#         if selected.size == 0:
#             # 发布默认值
#             msg = Float32MultiArray()
#             msg.data = [math.nan, math.nan, math.nan]
#             self.carriage_pub_.publish(msg)
            
#             type_msg = Int32()
#             type_msg.data = -1  # 表示未知车型
#             self.carriage_type_pub_.publish(type_msg)
#         else:
#             selected_float = selected.astype(np.float32)
#             diff = np.ptp(selected_float, axis=0)
            
#             # 发布尺寸信息
#             msg = Float32MultiArray()
#             msg.data = [float(diff[1]), float(diff[0]), float(diff[2])]
#             self.carriage_pub_.publish(msg)
            
#             # 识别车型并发布
#             carriage_type = self.identify_carriage_type([diff[1], diff[0], diff[2]])
#             type_msg = Int32()
#             type_msg.data = carriage_type
#             self.carriage_type_pub_.publish(type_msg)
            
#             self.get_logger().info(f'Detected carriage type: {carriage_type}, dimensions: {[diff[1], diff[0], diff[2]]}')


#     def identify_carriage_type(self, dimensions):
#         """根据尺寸识别车型"""
#         try:
#             # 使用KNN分类器预测车型
#             dimensions_array = np.array(dimensions).reshape(1, -1)
#             predicted_type = self.type_classifier.predict(dimensions_array)[0]
            
#             # 可选：添加距离阈值检查，如果距离太远则认为是未知车型
#             distances, indices = self.type_classifier.kneighbors(dimensions_array)
#             min_distance = distances[0][0]
            
#             # 设置一个距离阈值，如果最小距离超过阈值则认为不匹配
#             threshold = 1.0  # 这个阈值可以根据实际情况调整
#             if min_distance > threshold:
#                 return -1  # 未知车型
                
#             return int(predicted_type)
            
#         except Exception as e:
#             self.get_logger().error(f'Error in carriage type identification: {str(e)}')
#             return -1
        

#     def compare_result(self, xyz_arr, denormal_arr):
#         de_cols = denormal_arr[:, :3]
#         xyz_round = np.round(xyz_arr)
#         de_round = np.round(de_cols)
#         match_matrix = (de_round[:, np.newaxis, :] == xyz_round).all(axis=2)
#         mask = match_matrix.any(axis=1)

#         return denormal_arr[mask]


# def main(args=None):
#     rclpy.init(args=args)
#     rclpy.spin(CarriageNode('carriage_node'))
#     rclpy.shutdown()




#降采样+车型匹配

import os
import torch
import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import open3d as o3d
import threading
from queue import Queue
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float32MultiArray, Int32
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from python_algorithms.pointnet_part_seg_msg import PartSegMsgGetModel as pointnet2
from python_algorithms.carriage_node_utils import PartNormalDataset


class CarriageNode(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)

        # 订阅融合后的点云
        self.image_sub_ = self.create_subscription(PointCloud2, '/cloud_registration',
                                                   self.cloud_registration_callback,
                                                   qos)
        self.get_logger().info('subscribe /cloud_registration.')

        self.carriage_pub_ = self.create_publisher(Float32MultiArray, '/carriage_seg_res', 10)
        self.carriage_type_pub_ = self.create_publisher(Int32, '/carriage_type', 10)
        # 新增：发布采样后的点云
        self.sampled_cloud_pub_ = self.create_publisher(PointCloud2, '/sampled_pointcloud', 10)
        self.get_logger().info('publisher /carriage_seg_res, /carriage_type and /sampled_pointcloud.')

        # 加载模型
        self.part_model_ = pointnet2(num_classes=2, normal_channel=False).eval()
        share_pkg_dir = get_package_share_directory('python_algorithms')
        part_model_path = os.path.join(share_pkg_dir, 'models', 'part_seg.pth')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        weight_dict = torch.load(part_model_path, map_location=device, weights_only=False)
        self.part_model_.load_state_dict(weight_dict['model_state_dict'])
        self.get_logger().info('model part_seg loaded.')

        self.label_ = np.array([0]).astype(np.int32)

        # 车型映射表: [长, 宽, 高] 对应车型号
        self.carriage_type_mapping = {
            1: [4.2, 1.8, 1.6],   # 车型1的标准尺寸
            2: [5.0, 2.0, 1.8],   # 车型2的标准尺寸
            3: [6.5, 2.3, 2.2],   # 车型3的标准尺寸
            4: [8.0, 2.5, 2.5],   # 车型4的标准尺寸
            5: [13.5, 3.5, 1.5]   # 车型5的标准尺寸
        }
        
        # 使用KNN分类器来匹配最接近的车型
        self.type_classifier = self._init_type_classifier()
        
        # 处理队列和线程
        self.processing_queue = Queue(maxsize=3)  # 限制队列大小，避免积压
        self.processing_thread = threading.Thread(target=self._process_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 性能统计
        self.last_process_time = time.time()
        self.frame_count = 0
        self.processing = False
        
        # 缓存上一次的结果，用于快速发布
        self.last_valid_result = None
        self.last_valid_type = -1
        
        # 存储原始消息头，用于发布采样点云
        self.last_msg_header = None
        
        self.get_logger().info('Carriage node initialized with performance optimizations.')

        #采样点参数
        self.sampled_point_count = 8000

        #创建数据集采样点参数
        self.test_data_set = 5000

    def _init_type_classifier(self):
        """初始化车型分类器"""
        features = []
        labels = []
        
        for carriage_type, dimensions in self.carriage_type_mapping.items():
            features.append(dimensions)
            labels.append(carriage_type)
            
            # 添加一些噪声数据以增强鲁棒性
            for _ in range(5):
                noise = np.random.normal(0, 0.1, 3)
                noisy_features = np.array(dimensions) + noise
                features.append(noisy_features.tolist())
                labels.append(carriage_type)
        
        # 创建并训练KNN分类器
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(features, labels)
        
        self.get_logger().info('Carriage type classifier initialized.')
        return knn

    def cloud_registration_callback(self, msg: PointCloud2):
        """点云回调函数 - 只负责接收和入队"""
        current_time = time.time()
        
        # 限制处理频率，避免队列积压
        if current_time - self.last_process_time < 0.2:  # 最大5Hz处理，小于融合后的点云数据发布频率
            return
            
        if self.processing_queue.qsize() >= 2:  # 如果队列中有未处理的数据，跳过新数据
            return
            
        try:
            # 快速提取点云数据
            xyz_arr = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            
            # 保存消息头用于发布采样点云
            self.last_msg_header = msg.header
            
            # 点云下采样 - 大幅减少处理点数
            original_count = len(xyz_arr)
            if len(xyz_arr) > self.sampled_point_count:
                # 随机下采样到5000个点
                indices = np.random.choice(len(xyz_arr), self.sampled_point_count, replace=False)
                xyz_arr = xyz_arr[indices]
                sampled_count = len(xyz_arr)
                self.get_logger().debug(f'Downsampled pointcloud: {original_count} -> {sampled_count} points')
            else:
                sampled_count = original_count
            
            # 发布采样后的点云用于可视化
            self._publish_sampled_pointcloud(xyz_arr)
            
            # 放入处理队列
            self.processing_queue.put(xyz_arr, block=False)
            self.last_process_time = current_time
            
        except Exception as e:
            self.get_logger().warn(f'Error in callback: {str(e)}')

    def _publish_sampled_pointcloud(self, xyz_arr):
        """发布采样后的点云用于可视化检查"""
        try:
            # 创建PointCloud2消息
            header = self.last_msg_header
            header.stamp = self.get_clock().now().to_msg()  # 更新时间戳
            
            # 创建字段描述
            fields = [
                point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1)
            ]
            
            # 创建点云消息
            cloud_msg = point_cloud2.create_cloud(header, fields, xyz_arr)
            
            # 发布消息
            self.sampled_cloud_pub_.publish(cloud_msg)
            
        except Exception as e:
            self.get_logger().warn(f'Error publishing sampled pointcloud: {str(e)}')

    def _process_worker(self):
        """后台处理线程"""
        while True:
            try:
                xyz_arr = self.processing_queue.get(timeout=1.0)
                self._process_pointcloud(xyz_arr)
            except:
                # 超时或其他异常，继续循环
                continue

    def _process_pointcloud(self, xyz_arr):
        """处理点云数据"""
        self.processing = True
        try:
            start_time = time.time()
            
            # 创建点云并估计法线
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(xyz_arr)
            
            # 优化法线估计参数
            pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=20))
            
            # 创建数据集 - 减少采样点数
            test_data_set = PartNormalDataset(pc, self.test_data_set, False)  # 5000点采样
            
            data_loader = torch.utils.data.DataLoader(test_data_set, batch_size=1,
                                                      shuffle=False, num_workers=0,
                                                      drop_last=True)
            
            for batch_id, (points, label, centroid, m) in enumerate(data_loader):
                points = points.transpose(2, 1).to(self.device)
                xyz_feature_point = points[:, :6, :]

                # 模型推理
                with torch.no_grad():  # 禁用梯度计算，提高速度
                    seg_pred, _ = self.part_model_(points, self.to_categorical(label, 1))
                
                seg_pred = seg_pred.cpu().data.numpy()
                seg_pred = np.argmax(seg_pred, axis=-1)
                seg_pred = np.concatenate([np.asarray(xyz_feature_point.cpu()), seg_pred[:, None, :]],
                                          axis=1).transpose((0, 2, 1)).squeeze(0)
                normalized_coords = seg_pred[:, :3]
                attributes = seg_pred[:, 3:]
                original_coords = m * normalized_coords + centroid
                denormalized_array = np.hstack((original_coords, attributes))

                de_res = self.compare_result(xyz_arr, denormalized_array)
                self._publish_carriage_info(de_res)
                
                # # 性能统计
                # process_time = time.time() - start_time
                # self.frame_count += 1
                # if self.frame_count % 10 == 0:
                #     self.get_logger().info(f'Processing frequency: {1.0/process_time:.2f}Hz')
                
        except Exception as e:
            self.get_logger().error(f'Error in pointcloud processing: {str(e)}')
        finally:
            self.processing = False

    def compare_result(self, xyz_arr, denormal_arr):
        de_cols = denormal_arr[:, :3]
        xyz_round = np.round(xyz_arr)
        de_round = np.round(de_cols)
        match_matrix = (de_round[:, np.newaxis, :] == xyz_round).all(axis=2)
        mask = match_matrix.any(axis=1)

        return denormal_arr[mask]

    def to_categorical(self, y, num_classes):
        new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
        if y.is_cuda:
            return new_y.cuda()
        return new_y

    def _publish_carriage_info(self, pred_data):
        """发布车辆信息"""
        selected = pred_data[pred_data[:, 3] == 1, :3]
        
        if selected.size == 0:
            # 发布默认值
            msg = Float32MultiArray()
            msg.data = [math.nan, math.nan, math.nan]
            self.carriage_pub_.publish(msg)
            
            type_msg = Int32()
            type_msg.data = -1  # 表示未知车型
            self.carriage_type_pub_.publish(type_msg)
        else:
            selected_float = selected.astype(np.float32)
            diff = np.ptp(selected_float, axis=0)
            
            # 发布尺寸信息
            msg = Float32MultiArray()
            msg.data = [float(diff[1]), float(diff[0]), float(diff[2])]
            self.carriage_pub_.publish(msg)
            
            # 识别车型并发布
            carriage_type = self.identify_carriage_type([diff[1], diff[0], diff[2]])
            type_msg = Int32()
            type_msg.data = carriage_type
            self.carriage_type_pub_.publish(type_msg)
            
            # 缓存有效结果
            self.last_valid_result = [float(diff[1]), float(diff[0]), float(diff[2])]
            self.last_valid_type = carriage_type
            
            self.get_logger().info(f'Detected carriage type: {carriage_type}, dimensions: [{diff[1]:.2f}, {diff[0]:.2f}, {diff[2]:.2f}]')

    def identify_carriage_type(self, dimensions):
        """根据尺寸识别车型"""
        try:
            # 使用KNN分类器预测车型
            dimensions_array = np.array(dimensions).reshape(1, -1)
            predicted_type = self.type_classifier.predict(dimensions_array)[0]
            
            # 可选：添加距离阈值检查，如果距离太远则认为是未知车型
            distances, indices = self.type_classifier.kneighbors(dimensions_array)
            min_distance = distances[0][0]
            
            # 设置一个距离阈值，如果最小距离超过阈值则认为不匹配
            threshold = 1.0
            if min_distance > threshold:
                return -1
                
            return int(predicted_type)
            
        except Exception as e:
            self.get_logger().error(f'Error in carriage type identification: {str(e)}')
            return -1

    def destroy_node(self):
        """清理资源"""
        self.processing_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarriageNode('carriage_node')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()