#!/bin/bash

echo "开始卸载 sp_msgs 包..."

# 停止所有可能的 ROS2 节点
ros2 node list | grep sp_msgs | while read node; do
    ros2 node kill $node
done

# 删除文件
sudo rm -rf /usr/local/include/sp_msgs
sudo rm -rf /usr/include/sp_msgs
sudo rm -f /usr/local/lib/libsp_msgs__*
sudo rm -f /usr/lib/libsp_msgs__*
sudo rm -rf /usr/local/share/sp_msgs
sudo rm -rf /usr/share/sp_msgs
sudo rm -rf /opt/ros/*/share/sp_msgs
sudo rm -rf /opt/ros/*/include/sp_msgs

# 清理开发工作空间（如果有）
rm -rf ~/ros2_ws/install/sp_msgs
rm -rf ~/ros2_ws/build/sp_msgs

echo "卸载完成。"
echo "剩余的相关文件："
find /usr -name "*sp_msgs*" 2>/dev/null
find /opt -name "*sp_msgs*" 2>/dev/null