#include "process_point_cloud/utils.h"

#include <ament_index_cpp/get_package_share_directory.hpp>


YAML::Node LoadConfigFile(const std::string& file_path)
{
    std::string config_path;
    if (file_path.empty()) {
        using namespace ament_index_cpp;
        auto share_dir{get_package_share_directory("process_point_cloud")};
        config_path = share_dir + "/config/process_node_config.yaml";
    } else {
        config_path = file_path;
    }

    return YAML::LoadFile(config_path);
}

constexpr inline bool isLittleEndian()
{
    uint16_t num{0x0001};
    auto ptr{reinterpret_cast<uint8_t*>(&num)};

    return (ptr[0] == 0x01);
}

uint32_t LittleEndianToBigEndian(uint32_t num)
{
    if (!isLittleEndian()) {
        return num;
    }

    uint32_t big_endian_num{((num & 0xFF000000) >> 24)
                            | ((num & 0x00FF0000) >> 8)
                            | ((num & 0x0000FF00) << 8)
                            | ((num & 0x000000FF) << 24)};

    return big_endian_num;
}
