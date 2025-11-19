#ifndef PLC_COMMUNICATION_NODE_HPP
#define PLC_COMMUNICATION_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include "snap7.h"
#include <memory>
#include <string>

namespace plc_communication
{
class PLCCommunicationNode : public rclcpp::Node
{
public:
    PLCCommunicationNode(const rclcpp::NodeOptions & options);
    ~PLCCommunicationNode();

private:
    void heightCallback(const std_msgs::msg::Float32::SharedPtr msg);
    void areaCallback(const std_msgs::msg::Float32::SharedPtr msg);
    bool connectToPLC();
    void disconnectFromPLC();
    bool sendDataToPLC(double value,int offset);
    void checkConnection();
    
    // PLC连接相关
    std::shared_ptr<TS7Client> client_;  // 声明shared_ptr管理S7客户端
    std::string plc_ip_;
    int plc_rack_;
    int plc_slot_;
    int db_number_;
    int data_offset_height_;
    int data_offset_area_;
    int plc_data_type_;
    bool is_connected_ = false;
    
    // 数据范围
    double min_height_;
    double max_height_;
    
    // ROS2相关
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr height_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr area_sub_;
    rclcpp::TimerBase::SharedPtr connection_timer_;
    rclcpp::Time last_success_time_;
};

}// namespace plc_communication

#endif // PLC_COMMUNICATION_NODE_HPP