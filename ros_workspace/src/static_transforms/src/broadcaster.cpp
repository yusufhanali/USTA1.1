#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/static_transform_broadcaster.h"

#include <fstream>
#include <jsoncpp/json/json.h>

std::vector<std::vector<std::string>> parseTransformations(const std::string &transfile) {
    std::ifstream file(transfile, std::ifstream::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + transfile);
    }

    Json::Value root;
    file >> root;

    std::vector<std::vector<std::string>> transformations;
    for (const auto &transformation : root["transformations"]) {
        std::vector<std::string> t;
        t.push_back(transformation["parent_frame_id"].asString());
        t.push_back(transformation["child_frame_id"].asString());
        t.push_back(transformation["x"].asString());
        t.push_back(transformation["y"].asString());
        t.push_back(transformation["z"].asString());
        t.push_back(transformation["qx"].asString());
        t.push_back(transformation["qy"].asString());
        t.push_back(transformation["qz"].asString());
        t.push_back(transformation["qw"].asString());
        transformations.push_back(t);
    }

    return transformations;
}


class StaticFramePublisher : public rclcpp::Node
{
public:
  explicit StaticFramePublisher(const std::string &transfile)
  : Node("static_tf_broadcaster")
  {
    tf_static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

    // Publish static transforms once at startup
    this->make_transforms(transfile);
  }

private:
  void make_transforms(const std::string &transfile)
  {
    std::vector<std::vector<std::string>> transforms = parseTransformations(transfile);

    for (const auto &transformation : transforms) {
      geometry_msgs::msg::TransformStamped transform;
      transform.header.stamp = this->now();
      transform.header.frame_id = transformation[0];
      transform.child_frame_id = transformation[1];
      transform.transform.translation.x = std::stod(transformation[2]);
      transform.transform.translation.y = std::stod(transformation[3]);
      transform.transform.translation.z = std::stod(transformation[4]);
      transform.transform.rotation.x = std::stod(transformation[5]);
      transform.transform.rotation.y = std::stod(transformation[6]);
      transform.transform.rotation.z = std::stod(transformation[7]);
      transform.transform.rotation.w = std::stod(transformation[8]);
      RCLCPP_INFO(this->get_logger(), "Sending static transform from %s to %s", transformation[0].c_str(), transformation[1].c_str());
      tf_static_broadcaster_->sendTransform(transform);
    }
  }

  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;
};

int main(int argc, char * argv[])
{
  auto logger = rclcpp::get_logger("logger");

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StaticFramePublisher>("/home/kovan/USTA1.1/ros_workspace/src/static_transforms/src/transformations.json"));
  rclcpp::shutdown();
  return 0;
}
