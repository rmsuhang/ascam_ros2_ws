#include "process_point_cloud/process_node.h"
#include "process_point_cloud/utils.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <cv_bridge/cv_bridge.h>

#include <chrono>


ProcessPointCloudNode::ProcessPointCloudNode(const std::string &node_name)
    : Node{node_name}
    , kImageWidth(1000)
    , kSaveParamStr{"save_registration_cloud"}
    , s7client_{new TS7Client{}}
    , rslidar1_cloud_{new CloudData{}}
    , rslidar2_cloud_{new CloudData{}}
    , cloud1_ready_{false}
    , cloud2_ready_{false}
    , processed_carriage_{false}
    , already_stopped_{false}
    , save_cloud_{false}
    , last_rear_len_{NAN}
    , rslidar1_area_{}
    , rslidar2_area_{}
    , adjust_params_{}
    , stop_area_{}
{
    rslidar1_sub_ = create_subscription<CloudMsg>(
        "/rslidar_points1", rclcpp::SensorDataQoS{},
        std::bind(&ProcessPointCloudNode::rslidar1SubCallback,
                  this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "subscribe /rslidar_points1.");

    rslidar2_sub_ = create_subscription<CloudMsg>(
        "/rslidar_points2", rclcpp::SensorDataQoS{},
        std::bind(&ProcessPointCloudNode::rslidar2SubCallback,
                  this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "subscribe /rslidar_points2.");

    registration_pub_ = create_publisher<CloudMsg>(
        "/cloud_registration", rclcpp::QoS{10}.reliable());
    RCLCPP_INFO(get_logger(), "publisher /cloud_registration.");

    image_pub_ = create_publisher<ImageMsg>("/image_data", 10);
    RCLCPP_INFO(get_logger(), "publisher /image_data.");

    detect_sub_ = create_subscription<DetectionMsg>(
        "/detection_result", 10,
        std::bind(&ProcessPointCloudNode::detectSubCallback,
                  this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "subscribe /detection_result.");

    carriage_sub_ = create_subscription<FloatArrMsg>(
        "/carriage_seg_res", 10,
        std::bind(&ProcessPointCloudNode::carriageSegResSubCallback,
                  this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "subscribe /carriage_seg_res.");

    using namespace std::chrono_literals;
    timer_ = create_wall_timer(
        100ms, std::bind(&ProcessPointCloudNode::timerCallback, this));
    RCLCPP_INFO(get_logger(), "timer created.");

    loadConfigurations();
    RCLCPP_INFO(get_logger(), "load configurations.");

    declare_parameter(kSaveParamStr, std::string{"OFF"});
    auto res{add_on_set_parameters_callback(
        std::bind(&ProcessPointCloudNode::setParameterCallback,
                  this, std::placeholders::_1))};
    (void)res;
}

ProcessPointCloudNode::~ProcessPointCloudNode()
{
    if (s7client_->Connected()) {
        s7client_->Disconnect();
    }
}

void ProcessPointCloudNode::rslidar1SubCallback(const CloudMsgPtr msg)
{
    pcl::fromROSMsg(*msg, *rslidar1_cloud_);
    cloud1_ready_.store(true);
}

void ProcessPointCloudNode::rslidar2SubCallback(const CloudMsgPtr msg)
{
    pcl::fromROSMsg(*msg, *rslidar2_cloud_);
    cloud2_ready_.store(true);
}

void ProcessPointCloudNode::detectSubCallback(const DetectionMsgPtr msg)//
{
    bumper_map_.clear();//清空挡板位置数据
    for (const auto& detection : msg->detections) {//
        auto size_x{detection.bbox.size_x};
        auto size_y{detection.bbox.size_y};
        auto center_x{detection.bbox.center.position.x};
        auto center_y{detection.bbox.center.position.y};

        int x1{static_cast<int>(center_x - size_x / 2)};
        int y1{static_cast<int>(center_y - size_y / 2)};
        int x2{static_cast<int>(center_x + size_x / 2)};
        int y2{static_cast<int>(center_y + size_y / 2)};
        for (const auto& hypothesis : detection.results) {
            std::string class_id{hypothesis.hypothesis.class_id};
            auto score{hypothesis.hypothesis.score};
            if (score > 0.5f
                && (class_id == "front bumper" || class_id == "rear bumper")) {
                bumper_map_.insert(std::make_pair(class_id,
                                                  BumperCoords{x1, y1, x2, y2, score}));
            }
        }
    }

    if (already_stopped_ && !processed_carriage_) {//处理已停止状态下的点云数据
        RCLCPP_INFO(get_logger(), "processing the registration cloud data.");
        publishCloudMsg(registration_pub_, registration_cloud_);
        if (save_cloud_) {
            saveRegistrationCloudData();
        }
        processed_carriage_.store(true);
    }

    auto front_pos{identifyCarriageFrontPos()};//识别前挡板位置
    if (NAN != front_pos
        && std::abs(front_pos - stop_area_.stop_position) <= 0.02//位置差距在2cm以内，可以停车
        && !already_stopped_) {
        already_stopped_.store(true);
        processed_carriage_.store(false);
        RCLCPP_INFO(get_logger(), "send stop signals.");
        // DB44.DBX52.2 1-移车；0-停止移车
        if (!writeBoolToPlc(44, 52, 2, false))
        {
            RCLCPP_ERROR(get_logger(), "can not write data to DB44.DBX52.2.");
        }
    }

    auto rear_len{identifyCarriageRearDistance()};//识别车后挡板位置
    if (NAN != rear_len) {
        RCLCPP_INFO(get_logger(), "rear distance %f.", rear_len);
        // 车尾位置 DB45.DBD12
        if (!writeFloatToPlc(45, 12, rear_len)) {
            RCLCPP_ERROR(get_logger(), "can not write data to DB45.DBD12.");
        }
    }

    if (last_rear_len_ != NAN && std::abs(last_rear_len_ - rear_len) > 0.2) {//车辆发生了移动
        already_stopped_.store(false);
    }
    last_rear_len_ = rear_len;
}

void ProcessPointCloudNode::carriageSegResSubCallback(const FloatArrMsgPtr msg)//订阅到长宽高数据， 给plc通信
{
    if (NAN == msg->data[0] || NAN == msg->data[1] || NAN == msg->data[2]) {
        RCLCPP_ERROR(get_logger(), "carriage seg data is not valid.");
        return;
    }

    RCLCPP_INFO(get_logger(), "length: %f, width: %f, height: %f",
                msg->data[0], msg->data[1], msg->data[2]);

    // 车长 DB45.DBD0
    if (!writeFloatToPlc(45, 0, msg->data[0])) {
        RCLCPP_ERROR(get_logger(), "can not write data to DB45.DBD0.");
    }
    // 车宽 DB45.DBD4
    if (!writeFloatToPlc(45, 4, msg->data[1])) {
        RCLCPP_ERROR(get_logger(), "can not write data to DB45.DBD4.");
    }
    // 车高 DB45.DBD8
    if (!writeFloatToPlc(45, 8, msg->data[2])) {
        RCLCPP_ERROR(get_logger(), "can not write data to DB45.DBD8.");
    }
}

ProcessPointCloudNode::SetParamRes
ProcessPointCloudNode::setParameterCallback(const std::vector<rclcpp::Parameter>& params)//???????????????????????????????????????????????
{
    auto result{rcl_interfaces::msg::SetParametersResult()};
    result.successful = true;

    for (const auto& param : params) {
        if (param.get_name().find(kSaveParamStr) != std::string::npos) {
            auto val{param.as_string()};
            std::transform(val.begin(), val.end(), val.begin(),
                           [](unsigned char c) { return std::toupper(c); });
            if ("ON" == val) {
                save_cloud_.store(true);
                RCLCPP_INFO(get_logger(), "set program save registration cloud data.");
            } else if ("OFF" == val) {
                save_cloud_.store(false);
                RCLCPP_INFO(get_logger(), "set program not save registration cloud data.");
            }
        }
    }

    return result;
}

void ProcessPointCloudNode::loadConfigurations()
{
    auto config{LoadConfigFile()};

    try {
        rslidar1_area_.axis_x.min =
            config["rsliadr1_filter_area"]["x_axis"]["min"].as<double>();
        rslidar1_area_.axis_x.max =
            config["rsliadr1_filter_area"]["x_axis"]["max"].as<double>();

        rslidar1_area_.axis_y.min =
            config["rsliadr1_filter_area"]["y_axis"]["min"].as<double>();
        rslidar1_area_.axis_y.max =
            config["rsliadr1_filter_area"]["y_axis"]["max"].as<double>();

        rslidar1_area_.axis_z.min =
            config["rsliadr1_filter_area"]["z_axis"]["min"].as<double>();
        rslidar1_area_.axis_z.max =
            config["rsliadr1_filter_area"]["z_axis"]["max"].as<double>();

        rslidar2_area_.axis_x.min =
            config["rsliadr2_filter_area"]["x_axis"]["min"].as<double>();
        rslidar2_area_.axis_x.max =
            config["rsliadr2_filter_area"]["x_axis"]["max"].as<double>();

        rslidar2_area_.axis_y.min =
            config["rsliadr2_filter_area"]["y_axis"]["min"].as<double>();
        rslidar2_area_.axis_y.max =
            config["rsliadr2_filter_area"]["y_axis"]["max"].as<double>();

        rslidar2_area_.axis_z.min =
            config["rsliadr2_filter_area"]["z_axis"]["min"].as<double>();
        rslidar2_area_.axis_z.max =
            config["rsliadr2_filter_area"]["z_axis"]["max"].as<double>();
    } catch (...) {
        RCLCPP_ERROR(get_logger(), "rslidar1/rslidar2_filter_area has wrong type params.");
        rclcpp::shutdown();
    }

    connectToPlc(config);
    readAdjustParams(config);
    vanStopArea(config);
}

void ProcessPointCloudNode::connectToPlc(const YAML::Node& config)
{
    std::string ip_str{};
    int rack{};
    int slot{};
    try {
        ip_str = config["plc_connection"]["ip"].as<std::string>();
        rack = config["plc_connection"]["rack"].as<int>();
        slot = config["plc_connection"]["slot"].as<int>();
    } catch (...) {
        RCLCPP_ERROR(get_logger(), "plc_connection has wrong type params.");
        rclcpp::shutdown();
    }

    if (0 == s7client_->ConnectTo(ip_str.data(), rack, slot)) {
        RCLCPP_INFO(get_logger(), "connect to plc.");
    } else {
        RCLCPP_ERROR(get_logger(), "can not connect to plc.");
        //rclcpp::shutdown();
    }
}

void ProcessPointCloudNode::readAdjustParams(const YAML::Node& config)
{
    try {
        adjust_params_.co_distance =
            config["adjust_parameters"]["co_distance"].as<double>();
        adjust_params_.max_iteration =
            config["adjust_parameters"]["max_iteration"].as<int>();
        adjust_params_.trans_epsilon =
            config["adjust_parameters"]["trans_epsilon"].as<double>();
        adjust_params_.euc_epsilon =
            config["adjust_parameters"]["euc_epsilon"].as<double>();
    } catch (...) {
        RCLCPP_ERROR(get_logger(), "adjust_parameters has wrong type params.");
        rclcpp::shutdown();
    }
}

void ProcessPointCloudNode::vanStopArea(const YAML::Node& config)//加载停车区域参数
{
    try {
        stop_area_.stop_position = config["stop_position"].as<double>();
        stop_area_.chute_diameter = config["chute_diameter"].as<double>();
    } catch (...) {
        RCLCPP_ERROR(get_logger(), "stop_position/chute has wrong type params.");
        rclcpp::shutdown();
    }
}

void ProcessPointCloudNode::timerCallback()//完成点云配准
{
    if (!cloud1_ready_ || !cloud2_ready_) {
        return;
    }
    cloud1_ready_.store(false);
    cloud2_ready_.store(false);

    CloudDataPtr cloud1_voxel{new CloudData{}};
    passThroughFilter(rslidar1_cloud_, cloud1_voxel, kRslidar1, kXAxis);
    passThroughFilter(cloud1_voxel, cloud1_voxel, kRslidar1, kYAxis);
    passThroughFilter(cloud1_voxel, cloud1_voxel, kRslidar1, kZAxis);

    CloudDataPtr cloud2_voxel{new CloudData{}};
    passThroughFilter(rslidar2_cloud_, cloud2_voxel, kRslidar2, kXAxis);
    passThroughFilter(cloud2_voxel, cloud2_voxel, kRslidar2, kYAxis);
    passThroughFilter(cloud2_voxel, cloud2_voxel, kRslidar2, kZAxis);

    CloudDataPtr registration_cloud{new CloudData{}};
    performRegistration(cloud1_voxel, cloud2_voxel, registration_cloud);
    performFilter(registration_cloud, registration_cloud);

    CloudDataPtr yoz_cloud{new CloudData{}};
    projectionYOZ(registration_cloud, yoz_cloud);
    publishImageMsg(yoz_cloud);

    yoz_cloud_ = yoz_cloud;
    registration_cloud_ = registration_cloud;
}

//直通滤波
void ProcessPointCloudNode::passThroughFilter(CloudDataPtr cloud,
                                              CloudDataPtr &cloud_medium,
                                              RslidarNumber lidar,
                                              FilterAxes axis)
{
    pcl::PassThrough<pcl::PointXYZI> pass{};
    pass.setInputCloud(cloud);          // 待滤波点云

    // 滤波轴方向
    switch (axis) {
    case kXAxis:
        pass.setFilterFieldName("x");
        switch (lidar) {
        case kRslidar1:
            pass.setFilterLimits(rslidar1_area_.axis_x.min, rslidar1_area_.axis_x.max);
            break;
        case kRslidar2:
            pass.setFilterLimits(rslidar2_area_.axis_x.min, rslidar2_area_.axis_x.max);
            break;
        default:
            break;
        }
        break;
    case kYAxis:
        pass.setFilterFieldName("y");
        switch (lidar) {
        case kRslidar1:
            pass.setFilterLimits(rslidar1_area_.axis_y.min, rslidar1_area_.axis_y.max);
            break;
        case kRslidar2:
            pass.setFilterLimits(rslidar2_area_.axis_y.min, rslidar2_area_.axis_y.max);
            break;
        default:
            break;
        }
        break;
    case kZAxis:
        pass.setFilterFieldName("z");
        switch (lidar) {
        case kRslidar1:
            pass.setFilterLimits(rslidar1_area_.axis_z.min, rslidar1_area_.axis_z.max);
            break;
        case kRslidar2:
            pass.setFilterLimits(rslidar2_area_.axis_z.min, rslidar2_area_.axis_z.max);
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }

    pass.setNegative(false);            // false: 保留当前范围内数据, true: 取反
    pass.filter(*cloud_medium);         // 转存滤波结果
}

//ICP配准
void ProcessPointCloudNode::performRegistration(CloudDataPtr src_cloud,
                                                CloudDataPtr tgt_cloud,
                                                CloudDataPtr registration_cloud)
{
    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
    icp.setInputSource(src_cloud);
    icp.setInputTarget(tgt_cloud);

    // 设置参数
    icp.setMaxCorrespondenceDistance(adjust_params_.co_distance);
    icp.setMaximumIterations(adjust_params_.max_iteration);
    icp.setTransformationEpsilon(adjust_params_.trans_epsilon);
    icp.setEuclideanFitnessEpsilon(adjust_params_.euc_epsilon);

    icp.align(*registration_cloud);
    *registration_cloud = *registration_cloud + *tgt_cloud;

    if (!icp.hasConverged()) {
        RCLCPP_INFO(get_logger(), "ICP did not converge.");
    }
}

//
void ProcessPointCloudNode::performFilter(CloudDataPtr cloud, CloudDataPtr cloud_voxel)//？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
{
    // 矩形的最大和最小边界
    Eigen::Vector4f min_point{};
    min_point << 0.2 - 1.7 / 2, 1 - 1.5 / 2, 3.5 - 2 / 2, 1.0; // 最小边界点
    Eigen::Vector4f max_point{};
    max_point << 0.2 + 1.7 / 2, 1 + 1.5 / 2, 3.5 + 2 / 2, 1.0; // 最大边界点

    // 过滤器
    pcl::CropBox<pcl::PointXYZI> box_filter{};
    box_filter.setMin(min_point);
    box_filter.setMax(max_point);
    box_filter.setNegative(true); // 反向滤波，保留矩形框外侧的点
    box_filter.setInputCloud(cloud);
    box_filter.filter(*cloud_voxel); // 转存过滤后的点云
}

//投影到YOZ平面
void ProcessPointCloudNode::projectionYOZ(CloudDataPtr cloud, CloudDataPtr projection)
{
    // 投影到YOZ平面
    // 创建投影平面系数
    pcl::ModelCoefficients::Ptr coefficients{new pcl::ModelCoefficients};
    // x轴法向量投影在ros2中如何将沿x轴方向的长度为5的点云数据去除
    coefficients->values.resize(4);
    coefficients->values[0] = 1.0;
    coefficients->values[1] = 0.0;
    coefficients->values[2] = 0.0;
    coefficients->values[3] = 0.0; // 原点距离

    pcl::ProjectInliers<pcl::PointXYZI> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(cloud);
    proj.setModelCoefficients(coefficients);
    proj.filter(*projection);
}


void ProcessPointCloudNode::publishCloudMsg(CloudPubPtr pub, CloudDataPtr cloud)
{
    CloudMsgPtr cloud_msg{new CloudMsg{}};
    pcl::toROSMsg(*cloud, *cloud_msg);
    cloud_msg->header.frame_id = "rslidar";
    cloud_msg->header.stamp = now();
    pub->publish(*cloud_msg);
}

//发布yoz的点云图像消息
void ProcessPointCloudNode::publishImageMsg(CloudDataPtr cloud)
{
    // 创建图像，假设点云的范围在[0, 1]之间
    int width = 1000; // 图像宽度
    int height = 200; // 图像高度
    cv::Mat image{height, width, CV_8UC1, cv::Scalar(0)};

    pcl::PointXYZI min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);
    float range_y = max_pt.y - min_pt.y;
    float range_z = max_pt.z - min_pt.z;
    // 将点云转换为图像
    for (const auto &point : cloud->points) {
        int x = static_cast<int>((point.y - min_pt.y) / range_y * (width - 1));
        int y = static_cast<int>((point.z - min_pt.z) / range_z * (height - 1));

        if (x >= 0 && x < width && y >= 0 && y < height) {
            image.at<uchar>(y, x) = 255; // 将点映射到图像上，设置为白色
        }
    }

    // 垂直180度镜像
    cv::Mat rotated_image;
    cv::flip(image, rotated_image, 0);

    // 使用 cv_bridge 将 OpenCV 图像转换为 ROS2 图像消息
    std_msgs::msg::Header header;
    header.frame_id = "rslidar"; // 设置适当的坐标系
    header.stamp = this->now();
    ImageMsgPtr image_msg{cv_bridge::CvImage(header, "mono8", rotated_image).toImageMsg()};

    // 发布图像消息
    image_pub_->publish(*image_msg);
}

//计算前挡板位置
double ProcessPointCloudNode::identifyCarriageFrontPos()
{
    pcl::PointXYZI min_pt{};
    pcl::PointXYZI max_pt{};
    pcl::getMinMax3D(*yoz_cloud_, min_pt, max_pt);
    auto rng_y{max_pt.y - min_pt.y};

    auto front_it{bumper_map_.find("front bumper")};
    if (front_it != bumper_map_.end()) {
        return (static_cast<float>(front_it->second.x1)
                / (kImageWidth - 1)) * rng_y + min_pt.y;
    }

    return NAN;
}

//计算后挡板距离
double ProcessPointCloudNode::identifyCarriageRearDistance()
{
    pcl::PointXYZI min_pt{};
    pcl::PointXYZI max_pt{};
    pcl::getMinMax3D(*yoz_cloud_, min_pt, max_pt);
    auto rng_y{max_pt.y - min_pt.y};

    auto rear_it{bumper_map_.find("rear bumper")};
    if (rear_it != bumper_map_.end()) {
        auto rear_pos{(static_cast<float>(rear_it->second.x1)
                       / (kImageWidth - 1)) * rng_y + min_pt.y};
        return static_cast<float>(rear_pos - min_pt.y);
    }

    return NAN;
}

//保存配准点云数据
void ProcessPointCloudNode::saveRegistrationCloudData()
{
    auto now{std::chrono::system_clock::now()};
    auto now_ms{std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch())};
    auto now_ms_str{std::to_string(now_ms.count())};

    std::string file_dir{"/root/app/save_data/process_point_cloud_node/"};
    auto file_name = now_ms_str + ".pcd";

    pcl::io::savePCDFileASCII(file_dir + file_name, *registration_cloud_);
    RCLCPP_INFO(get_logger(), "saved registration %s.", file_name.c_str());
}

//写float到PLC
bool ProcessPointCloudNode::writeFloatToPlc(int db_num, int start_off, float value)
{
    if (!s7client_->Connected()) {
        return false;
    }

    Converter con{};
    con.f = value;
    auto be_value{LittleEndianToBigEndian(con.u)};
    uint8_t buffer[4]{};
    memcpy(buffer, &be_value, sizeof(be_value));
    auto ret{s7client_->WriteArea(S7AreaDB, db_num, start_off, 1, S7WLDWord, buffer)};

    return (0 == ret);
}

//写bool到PLC
bool ProcessPointCloudNode::writeBoolToPlc(int db_num, int start_off,
                                           int bit_off, bool status)
{
    if (!s7client_->Connected()) {
        return false;
    }

    uint8_t byte_data{};
    auto ret{s7client_->ReadArea(S7AreaDB, db_num, start_off, 1, S7WLByte, &byte_data)};
    if (0 != ret) {
        return false;
    }

    if (status) {
        byte_data |= (1 << bit_off);
    } else {
        byte_data &= ~(1 << bit_off);
    }
    ret = s7client_->WriteArea(S7AreaDB, db_num, start_off, 1, S7WLByte, &byte_data);

    return (0 == ret);
}
