#include <yaml-cpp/yaml.h>


YAML::Node LoadConfigFile(const std::string& file_path = std::string{});

uint32_t LittleEndianToBigEndian(uint32_t num);
