# 定量装车项目

## project environment
    1. ros2 humble：提供项目框架

    2. anaconda3+py3.10+pytorch+yolov5（版本不要太高）：处理深度学习算法部分

    3. sanp7：plc通信部分
    
    4. others：运行过程中缺少什么安装什么即可

## package overview

    1. calibrate_point_cloud：点云配准算法，使用icp算法，输出一个刚体变换，使源点云和目标点云重合度最高

    2. baffle_node：根据配准后的点云图像进行挡板检测，前挡板，后挡板，下料口

    3. detection_bbox_node：根据识别结果，计算并更新配置文件中的挡板位置参数

    4. calibrate_launch：launch文件，启动1，2，3节点

    5. process_point_cloud：整合了1,3节点和plc通信

    6. python_algorithms:python算法集合

            ├──baffle_node：挡板检测

            ├──carriage_node：车厢分割+尺寸检测

            ├──carriage_node_utils：车厢分割工具类

            ├──pointnet_part_seg_msg：pointnet算法
            
            ├──pointnet2_utils：pointnet算法工具类
                     
    7. process_cloud_launch：launch文件，启动5，6相关节点

## usage：
 
    1. 算法测试（不跟plc通信）：运行calibrate_launch包，完成点云融合配准，运行process_cloud_launch包，只带起carriage_node一个节点

    2. 完整部署：运行process_cloud_launch包，带起process_node, baffle_node, carriage_node三个节点