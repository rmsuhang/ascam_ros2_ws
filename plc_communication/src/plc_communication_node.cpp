#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>
#include <cstring>  // 添加memcpy支持

#include "plc_communication/plc_communication_node.hpp"

namespace plc_communication
{

PLCCommunicationNode::PLCCommunicationNode(const rclcpp::NodeOptions & options)
    : Node("plc_communication_node", options)
{
    this->declare_parameter<std::string>("plc_ip", "192.168.2.1");
    this->declare_parameter<int>("plc_rack", 0);
    this->declare_parameter<int>("plc_slot", 2);
    this->declare_parameter<int>("db_number", 1);
    this->declare_parameter<int>("data_offset_height",700);
    this->declare_parameter<int>("data_offset_length", 704);//面积数据偏移地址
    this->declare_parameter<int>("data_offset_width", 708);//面积数据偏移地址

    this->declare_parameter<double>("min_height", 0.0);
    this->declare_parameter<double>("max_height", 1.0);
    this->declare_parameter<int>("plc_data_type", 0);  // 0:REAL, 1:INT, 2:DINT
    
    plc_ip_ = this->get_parameter("plc_ip").as_string();
    plc_rack_ = this->get_parameter("plc_rack").as_int();
    plc_slot_ = this->get_parameter("plc_slot").as_int();
    db_number_ = this->get_parameter("db_number").as_int();
    data_offset_height_ = this->get_parameter("data_offset_height").as_int();
    data_offset_length_ = this->get_parameter("data_offset_length").as_int();
    data_offset_width_ = this->get_parameter("data_offset_width").as_int();
    min_height_ = this->get_parameter("min_height").as_double();
    max_height_ = this->get_parameter("max_height").as_double();
    plc_data_type_ = this->get_parameter("plc_data_type").as_int();
    
    // 创建S7客户端
    client_ = std::make_shared<TS7Client>();//创建shared_ptr管理的S7客户端
    client_-> SetConnectionType(3);

    // 订阅高度差数据
    height_sub_ = this->create_subscription<std_msgs::msg::Float32>(
        "/features/height_statistics",
        rclcpp::QoS(rclcpp::KeepLast(10)),
        std::bind(&PLCCommunicationNode::heightCallback, this, std::placeholders::_1));

    // 订阅边界框尺寸数据
    bbox_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
        "/features/bounding_box_dimensions",
        rclcpp::QoS(rclcpp::KeepLast(10)),
        std::bind(&PLCCommunicationNode::bboxCallback, this, std::placeholders::_1));

        
    // 连接PLC
    if (connectToPLC()) {
        RCLCPP_INFO(this->get_logger(), "Successfully connected to PLC at %s", plc_ip_.c_str());
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to connect to PLC at %s", plc_ip_.c_str());
    }
    
    // 创建定时器检查连接状态
    connection_timer_ = this->create_wall_timer(
        std::chrono::seconds(5),
        std::bind(&PLCCommunicationNode::checkConnection, this));
    
    RCLCPP_INFO(this->get_logger(), "PLC Communication node initialized");
}

PLCCommunicationNode::~PLCCommunicationNode()
{
    disconnectFromPLC();
}

void PLCCommunicationNode::heightCallback(const std_msgs::msg::Float32::SharedPtr msg)
{
    double height_value = msg->data;

    // 发送数据到PLC
    if (sendDataToPLC(height_value, data_offset_height_)) {
        RCLCPP_INFO(this->get_logger(), 
                    "Successfully sent height value %.4f to PLC at offset %d", 
                    height_value, data_offset_height_);
    } else {
        RCLCPP_INFO(this->get_logger(), 
                    "Failed to send height value %.4f to PLC at offset %d", 
                    height_value, data_offset_height_);
    }

    
}


void PLCCommunicationNode::bboxCallback(const geometry_msgs::msg::Vector3::SharedPtr msg)
{
    double length_value = msg->x;  // 长度数据
    double width_value = msg->y;   // 宽度数据

    // 发送长度数据到PLC
    if (sendDataToPLC(length_value, data_offset_length_)) {
        RCLCPP_INFO(this->get_logger(), 
                    "Successfully sent length value %.6f to PLC at offset %d", 
                    length_value, data_offset_length_);
    } else {
        RCLCPP_INFO(this->get_logger(), 
                    "Failed to send length value %.6f to PLC at offset %d", 
                    length_value, data_offset_length_);
    }

    // 发送宽度数据到PLC
    if (sendDataToPLC(width_value, data_offset_width_)) {
        RCLCPP_INFO(this->get_logger(), 
                    "Successfully sent width value %.6f to PLC at offset %d", 
                    width_value, data_offset_width_);
    } else {
        RCLCPP_INFO(this->get_logger(), 
                    "Failed to send width value %.6f to PLC at offset %d", 
                    width_value, data_offset_width_);
    }

}

// 连接到PLC
bool PLCCommunicationNode::connectToPLC()
{
    if (!client_) {
        RCLCPP_ERROR(this->get_logger(), "S7 client not initialized");
        return false;
    }
    
    // 连接到PLC
    int result = client_->ConnectTo(plc_ip_.c_str(), plc_rack_, plc_slot_);
    if (result == 0) {
        is_connected_ = true;
        return true;
    } else {
        RCLCPP_ERROR(this->get_logger(), 
                    "Failed to connect to PLC: error code %d", result);
        is_connected_ = false;
        return false;
    }
}

// 断开与PLC的连接
void PLCCommunicationNode::disconnectFromPLC()
{
    if (client_ && is_connected_) {
        client_->Disconnect();
        is_connected_ = false;
        RCLCPP_INFO(this->get_logger(), "Disconnected from PLC");
    }
}

// 发送数据到PLC
bool PLCCommunicationNode::sendDataToPLC(double value,int offset)
{
    if (!is_connected_ || !client_) {
        RCLCPP_WARN(this->get_logger(), "Not connected to PLC, attempting to reconnect");
        if (!connectToPLC()) {
            return false;
        }
    }
    
    int result = 0;
    
    // 根据数据类型准备并发送数据
    switch (plc_data_type_) {
        case 0: {  // REAL (4 bytes)
            float real_value = static_cast<float>(value);
            result = client_->DBWrite(db_number_, offset, sizeof(float), &real_value);
            break;
        }
        case 1: {  // INT (2 bytes)
            int16_t int_value = static_cast<int16_t>(value * 1000);  // 转换为毫米
            result = client_->DBWrite(db_number_, offset, sizeof(int16_t), &int_value);
            break;
        }
        case 2: {  // DINT (4 bytes)
            int32_t dint_value = static_cast<int32_t>(value * 1000);  // 转换为毫米
            result = client_->DBWrite(db_number_, offset, sizeof(int32_t), &dint_value);
            break;
        }
        default:
            RCLCPP_ERROR(this->get_logger(), "Unknown PLC data type: %d", plc_data_type_);
            return false;
    }
    if (result == 0) {
        last_success_time_ = this->now();
        return true;
    } else {
        RCLCPP_ERROR(this->get_logger(), 
                    "Failed to write data to PLC DB%d at offset %d: error %d", db_number_, offset, result);
        is_connected_ = false;  // 标记连接断开
        return false;
    }
}

// 检查PLC连接状态
void PLCCommunicationNode::checkConnection()
{
    if (!client_) {
        RCLCPP_ERROR(this->get_logger(), "S7 client not initialized");
        return;
    }
    
    if (!is_connected_) {
        RCLCPP_WARN(this->get_logger(), "PLC connection lost, attempting to reconnect");
        connectToPLC();
        return;
    }
    
    bool connection_status = client_->Connected();
    if (!connection_status) {
        RCLCPP_WARN(this->get_logger(), "PLC connection check failed");
        is_connected_ = false;
    }
    
    // 定期输出
    static int counter = 0;
    if (++counter % 12 == 0) {  // 每分钟输出一次
        RCLCPP_INFO(this->get_logger(), 
                    "PLC connection status: %s", 
                    is_connected_ ? "Connected" : "Disconnected");
        counter = 0;
    }
}

}  // namespace plc_communication

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(plc_communication::PLCCommunicationNode)