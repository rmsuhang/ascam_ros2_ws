#include "process_point_cloud/snap7.h"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


namespace YAML {
    class Node;
} // YAML
class ProcessPointCloudNode : public rclcpp::Node {
public:
    ProcessPointCloudNode(const std::string& node_name);
    ~ProcessPointCloudNode();

    ProcessPointCloudNode(const ProcessPointCloudNode&) = delete;
    ProcessPointCloudNode& operator=(const ProcessPointCloudNode&) = delete;
    ProcessPointCloudNode(ProcessPointCloudNode&&) = delete;
    ProcessPointCloudNode& operator=(ProcessPointCloudNode&&) = delete;

private:
    using CloudMsg = sensor_msgs::msg::PointCloud2;
    using CloudMsgPtr = CloudMsg::SharedPtr;
    using CloudSubPtr = rclcpp::Subscription<CloudMsg>::SharedPtr;
    using CloudPubPtr = rclcpp::Publisher<CloudMsg>::SharedPtr;

    using ImageMsg = sensor_msgs::msg::Image;
    using ImageMsgPtr = ImageMsg::SharedPtr;
    using ImagePubPtr = rclcpp::Publisher<ImageMsg>::SharedPtr;

    using DetectionMsg = vision_msgs::msg::Detection2DArray;
    using DetectionMsgPtr = DetectionMsg::SharedPtr;
    using DetectionSubPtr = rclcpp::Subscription<DetectionMsg>::SharedPtr;

    using FloatArrMsg = std_msgs::msg::Float32MultiArray;
    using FloatArrMsgPtr = FloatArrMsg::SharedPtr;
    using FloatArrSubPtr = rclcpp::Subscription<FloatArrMsg>::SharedPtr;

    using CloudData = pcl::PointCloud<pcl::PointXYZI>;
    using CloudDataPtr = CloudData::Ptr;

    using SetParamRes = rcl_interfaces::msg::SetParametersResult;

    enum RslidarNumber {
        kRslidar1 = 0,
        kRslidar2
    };

    enum FilterAxes {
        kXAxis = 0,
        kYAxis,
        kZAxis
    };

    union Converter {
        float f;
        uint32_t u;
    };

    struct AxisRng {
        double min;
        double max;
    };

    struct FilterArea {
        AxisRng axis_x;
        AxisRng axis_y;
        AxisRng axis_z;
    };

    struct AdjustParams {
        double co_distance;
        int max_iteration;
        double trans_epsilon;
        double euc_epsilon;
    };

    struct CarStopArea {
        double stop_position;
        double chute_diameter;
    };
    struct BumperCoords {
        int x1;
        int y1;
        int x2;
        int y2;
        double score;
    };

    void rslidar1SubCallback(const CloudMsgPtr msg);
    void rslidar2SubCallback(const CloudMsgPtr msg);
    void detectSubCallback(const DetectionMsgPtr msg);
    void carriageSegResSubCallback(const FloatArrMsgPtr msg);
    SetParamRes setParameterCallback(const std::vector<rclcpp::Parameter>& params);

    void loadConfigurations();
    void connectToPlc(const YAML::Node& config);
    void readAdjustParams(const YAML::Node& config);
    void vanStopArea(const YAML::Node& config);

    void timerCallback();

    // 直通滤波
    void passThroughFilter(CloudDataPtr cloud,
                           CloudDataPtr& cloud_medium,
                           RslidarNumber lidar,
                           FilterAxes axis);

    // 配准
    void performRegistration(CloudDataPtr src_cloud,
                             CloudDataPtr tgt_cloud,
                             CloudDataPtr registration_cloud);
    void performFilter(CloudDataPtr cloud, CloudDataPtr cloud_voxel);

    void projectionYOZ(CloudDataPtr cloud, CloudDataPtr projection);
    void publishCloudMsg(CloudPubPtr pub, CloudDataPtr cloud);
    void publishImageMsg(CloudDataPtr cloud);

    double identifyCarriageFrontPos();
    double identifyCarriageRearDistance();

    void saveRegistrationCloudData();

    // PLC
    bool writeFloatToPlc(int db_num, int start_off, float value);
    bool writeBoolToPlc(int db_num, int start_off, int bit_off, bool status);

    const int                       kImageWidth;
    const std::string               kSaveParamStr;

    std::unique_ptr<TS7Client>      s7client_;

    CloudDataPtr                    rslidar1_cloud_;
    CloudDataPtr                    rslidar2_cloud_;
    CloudDataPtr                    registration_cloud_;
    CloudDataPtr                    yoz_cloud_;
    std::atomic_bool                cloud1_ready_;
    std::atomic_bool                cloud2_ready_;
    std::atomic_bool                processed_carriage_;
    std::atomic_bool                already_stopped_;
    std::atomic_bool                save_cloud_;
    double                          last_rear_len_;

    FilterArea                      rslidar1_area_;
    FilterArea                      rslidar2_area_;
    AdjustParams                    adjust_params_;
    CarStopArea                     stop_area_;

    CloudSubPtr                     rslidar1_sub_;
    CloudSubPtr                     rslidar2_sub_;
    CloudPubPtr                     registration_pub_;
    ImagePubPtr                     image_pub_;
    DetectionSubPtr                 detect_sub_;
    FloatArrSubPtr                  carriage_sub_;

    rclcpp::TimerBase::SharedPtr    timer_;

    std::map<std::string, BumperCoords> bumper_map_;
};
