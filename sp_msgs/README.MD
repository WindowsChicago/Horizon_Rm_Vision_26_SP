## SP_MSGS

by ZYL

#### 通过io/ros2部分的代码逆转写出的通信包，通过验证可以使用

## 安装和使用说明

### 安装到系统

**方法1：CMAKE编译安装 sp_msgs 包：**

```bash
cd sp_msgs
mkdir -p build && cd build
cmake ..
make
sudo make install
```

然后在该目录下搜索：“local_setup.sh”，找到这个文件，复制到主目录的build下，编译主程序

```
cmake ..
make -j16
sudo make install
```

编译完成后需要source才能正常使用ros部分组件

```
cd ..
source build/local_setup.sh
./build/publish_test
```

运行无报错即证明集成成功，以后每次**运行**需要ros部分都需要在终端source这个sh文件

**方法2：使用 colcon编译调用sp_msgs 包：**

```bash
# 假设 workspace 结构如下：
# sp_msgs/
#   src/
#     msg/  (您创建的包)

cd sp_msgs
colcon build --packages-select sp_msgs
source install/setup.bash
```

之后回到主文件夹的build下打开终端,先source一下setup.bash(注意相对位置)，之后编译才能正确引入sp_msgs包：

```
source ../sp_msgs/install/setup.bash
cmake ..
make -j16
./build/publish_test
```

运行无报错即证明集成成功，以后每次**编译和运行**时需要ros部分都需要在终端source这个sh文件

总的来说还是方法1相对方便些

## 验证安装

无论按方法1还是2,source后，可以通过以下命令验证消息是否可用：

```bash
# 在终端中运行
ros2 interface show sp_msgs/msg/EnemyStatusMsg
ros2 interface show sp_msgs/msg/AutoaimTargetMsg
```

有正常反馈即安装成功

## 卸载方法

要卸载通过方案1安装的 `sp_msgs` 包，您需要手动删除安装的文件，以下是详细步骤：

## 1. 查找安装的文件

首先找出 `sp_msgs` 包安装的位置：

```bash
# 查找头文件
find /usr -name "sp_msgs" -type d 2>/dev/null

# 查找库文件
find /usr -name "*sp_msgs*" -type f 2>/dev/null

# 查找 CMake 配置文件
find /usr -name "*sp_msgs*Config.cmake" -o -name "*sp_msgs*config.cmake" 2>/dev/null

# 查找 pkg-config 文件
find /usr -name "*sp_msgs*.pc" 2>/dev/null
```

## 2. 删除安装的文件

根据找到的文件路径，执行删除命令。通常安装在以下位置：

```bash
# 删除头文件（通常在 /usr/local/include 或 /usr/include）
sudo rm -rf /usr/local/include/sp_msgs
sudo rm -rf /usr/include/sp_msgs

# 删除库文件（通常在 /usr/local/lib 或 /usr/lib）
sudo rm -f /usr/local/lib/libsp_msgs__*
sudo rm -f /usr/lib/libsp_msgs__*
sudo rm -f /usr/local/lib/python3*/site-packages/sp_msgs*

# 删除 CMake 配置文件
sudo rm -f /usr/local/lib/cmake/sp_msgs/*
sudo rm -f /usr/lib/cmake/sp_msgs/*
sudo rm -rf /usr/local/share/sp_msgs
sudo rm -rf /usr/share/sp_msgs

# 删除 ROS2 相关文件（如果安装在 ROS2 目录）
sudo rm -rf /opt/ros/*/share/sp_msgs
sudo rm -rf /opt/ros/*/include/sp_msgs
```

## 3. 如果使用 colcon 安装

如果您使用了 colcon 安装到 ROS2 工作空间：

```bash
# 进入工作空间
cd ~/ros2_ws

# 卸载包
colcon build --packages-select sp_msgs --cmake-target uninstall

# 或者直接删除安装目录
rm -rf install/sp_msgs
rm -rf build/sp_msgs
```

## 4. 验证卸载

验证 `sp_msgs` 是否已完全卸载：

```bash
# 检查是否还能找到相关文件
find /usr -name "*sp_msgs*" 2>/dev/null
find /opt -name "*sp_msgs*" 2>/dev/null

# 检查 ROS2 是否还能找到包
ros2 pkg list | grep sp_msgs

# 在您的项目中重新配置，应该会显示找不到 sp_msgs
cd /path/to/your/project
mkdir -p build && cd build
cmake ..
```

## 5. 清理环境变量

如果修改了环境变量，请清理：

```bash
# 检查当前 shell 的环境变量
echo $CMAKE_PREFIX_PATH
echo $ROS_PACKAGE_PATH

# 从 ~/.bashrc 或 ~/.zshrc 中移除相关路径
# 编辑对应的 shell 配置文件，删除包含 sp_msgs 的行
```

## 完整的清理脚本

您可以创建一个清理脚本 `uninstall_sp_msgs.sh`：

```bash
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
```

给脚本执行权限并运行：
```bash
chmod +x uninstall_sp_msgs.sh
./uninstall_sp_msgs.sh
```

完成这些步骤后，`sp_msgs` 包应该就从系统中完全移除了。