# 自动测量水泥塌落度项目

## project environment
    1. ros2 humble：提供项目框架

    2. sanp7：plc通信部分
    
    3. others：运行过程中缺少什么安装什么即可

## package overview

    1. ascamera：官方相机包，发布rgb，depth，pointcloud

    2. pointcloud_correct：点云坐标变换节点，修正相机俯角带来的影响

    3. feature_extraction：特征提取节点，根据深度信息，提取水泥，计算水泥塌落度，延展度

    4. plc_communication：plc通信节点，订阅塌落度和延展度数据，发送给plc

    5. bring_up：整个项目的launch文件，集合了各个节点的config，负责顺序启动各个节点


## usage：
 
    1. 完整部署：运行bring_up包，带起ascamera, pointcloud_correct, feature_extraction，plc_communication四个节点

## notice：
    
    1. 在使用snap7与西门子smart-200系列通信时，rack_number设置为1,同时要用执行 SetConnectionType(3)函数，更改默认连接类型

    2. 如果采用网线直连plc，需手动设置上位机ip和plc处于同一网段，并且将plc的ip设置为网关，

        例如plc的ip为192.168.2.1 ：

            则上位机的ip设置为192.168.2.1～192.168.2.255任意一个

            子网掩码设置为255.255.255.0

            网关设置为192.168.2.1（plc的ip地址）

            然后ping一下plc的ip地址，看能否建立连接，能够ping通即说明成功

    3. 如果有wifi只需要上位机和plc都连接到wifi，然后ping一下plc的ip，跳过手动设置ip的部分，自动获取ip即可