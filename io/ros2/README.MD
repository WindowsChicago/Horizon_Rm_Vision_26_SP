## 整体架构分析

这是一个基于ROS2的自瞄视觉系统通信模块，主要实现视觉系统与导航系统之间的数据交换。

## 各文件详细分析

### 1. **CMakeLists.txt** - 构建配置
```cmake
# 关键配置：
- 静态库目标：io
- ROS2依赖：rclcpp, std_msgs, sp_msgs
- 条件编译：仅在找到ROS2环境时编译ROS2相关代码
- 源文件：ros2/publish2nav.cpp, ros2/ros2.cpp, ros2/subscribe2nav.cpp
```

### 2. **publish2nav.hpp/cpp** - 数据发布器
**功能**：将视觉系统检测到的目标位置发布给导航系统

**核心类**：`Publish2Nav`
```cpp
// 发布话题：auto_aim_target_pos
// 消息类型：std_msgs::msg::String
// 数据格式：x,y,z,w (逗号分隔的字符串)
```

**输入**：`Eigen::Vector4d target_pos`
- 包含4个double值的向量，表示目标3D位置和额外参数

**输出**：ROS2话题消息
- 格式：`"x,y,z,w"` 字符串
- 话题：`/auto_aim_target_pos`

### 3. **subscribe2nav.hpp/cpp** - 数据订阅器
**功能**：从导航系统接收敌方状态和自瞄目标信息

**核心类**：`Subscribe2Nav`
```cpp
// 订阅两个话题：
// 1. /enemy_status - 敌方状态信息
// 2. /autoaim_target - 自瞄目标信息
```

**输入**：ROS2话题消息
- `/enemy_status`：`sp_msgs/EnemyStatusMsg`
- `/autoaim_target`：`sp_msgs/AutoaimTargetMsg`

**输出**：
- `std::vector<int8_t> enemy_ids` - 不可击败的敌方ID列表
- `std::vector<int8_t> target_ids` - 自瞄目标ID列表

### 4. **ros2.hpp/cpp** - ROS2管理器
**功能**：统一管理发布者和订阅者的生命周期

**核心类**：`ROS2`
```cpp
// 管理功能：
- 初始化ROS2上下文
- 创建发布者和订阅者线程
- 提供统一的数据收发接口
```

### 5. **sp_msgs包** - 自定义消息定义
**消息类型**：
- `EnemyStatusMsg.msg`：敌方状态消息
  ```msg
  builtin_interfaces/Time timestamp
  int8[] invincible_enemy_ids
  ```
- `AutoaimTargetMsg.msg`：自瞄目标消息
  ```msg
  builtin_interfaces/Time timestamp  
  int8[] target_ids
  ```

## 数据流向分析

### 正向数据流（视觉→导航）
```
视觉算法 → Eigen::Vector4d → Publish2Nav → std_msgs::String → /auto_aim_target_pos → 导航系统
```

### 反向数据流（导航→视觉）
```
导航系统 → sp_msgs/EnemyStatusMsg → /enemy_status → Subscribe2Nav → vector<int8_t> → 视觉算法
导航系统 → sp_msgs/AutoaimTargetMsg → /autoaim_target → Subscribe2Nav → vector<int8_t> → 视觉算法
```

## 核心特性

### 1. **线程安全设计**
- 每个节点运行在独立线程中
- 使用线程安全队列存储接收到的消息
- 避免数据竞争和阻塞

### 2. **超时清理机制**
```cpp
// 接收两次消息后启动1500ms定时器
// 超时后清空队列，防止使用过期数据
```

### 3. **错误处理**
- ROS2环境检测和条件编译
- 库文件存在性检查
- 多架构支持（x86_64/aarch64）

### 4. **性能优化**
- 队列容量限制为1，只保留最新消息
- 后台线程处理ROS2通信，不阻塞主视觉线程

## 输入输出

### **系统输入**
1. **视觉系统输入**：
   - `Eigen::Vector4d target_pos` - 检测到的目标位置

2. **导航系统输入**：
   - `/enemy_status` - 敌方状态信息
   - `/autoaim_target` - 自瞄目标信息

### **系统输出**
1. **视觉系统输出**：
   - `std::vector<int8_t>` - 敌方ID列表
   - `std::vector<int8_t>` - 目标ID列表

2. **导航系统输出**：
   - `/auto_aim_target_pos` - 目标位置字符串

