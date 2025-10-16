#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <iomanip>
#include <sstream>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include "tools/thread_safe_queue.hpp"

namespace io
{
struct __attribute__((packed)) GimbalToVision
{
  uint8_t head = 0xCD;        // 包头 0xCD
  float pitch;                // 四字节 pitch
  float yaw;                  // 四字节 yaw
  uint8_t mode;               // 一字节 mode
  uint32_t timestamp;         // 四字节时间戳
  uint8_t bullet_speed;       // 一字节弹速
  uint8_t tail = 0xDC;        // 包尾 0xDC
    // float yaw_vel;
    // float pitch_vel;
};

struct __attribute__((packed)) VisionToGimbal
{
  uint8_t head = 0xCD;        // 包头 0xCD
  float pitch;                // 四字节 pitch
  float yaw;                  // 四字节 yaw
  uint8_t mode;               // 一字节 mode
  uint32_t timestamp;         // 四字节时间戳
  uint8_t tail = 0xDC;        // 包尾 0xDC
    // float yaw_vel;
    // float yaw_acc;
    // float pitch_vel;
    // float pitch_acc;
};

static_assert(sizeof(VisionToGimbal) <= 64);

enum class GimbalMode
{
  IDLE,        // 空闲
  AUTO_AIM,    // 自瞄
  SMALL_BUFF,  // 小符
  BIG_BUFF     // 大符
};

struct GimbalState
{
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float bullet_speed;
  uint16_t bullet_count;
};

class Gimbal
{
public:
  Gimbal(const std::string & config_path);
  ~Gimbal();

  GimbalMode mode() const;
  GimbalState state() const;
  std::string str(GimbalMode mode) const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  void send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);

  void send(io::VisionToGimbal VisionToGimbal);

  // 数据包转换为十六进制字符串调试
  std::string packet_to_hex(const void* data, size_t size) const;

private:
  int fd_ = -1;
  std::string com_port_;
  
  std::thread thread_;
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;

  GimbalToVision rx_data_;
  VisionToGimbal tx_data_;

  GimbalMode mode_ = GimbalMode::IDLE;
  //GimbalMode mode_ = GimbalMode::AUTO_AIM;
  GimbalState state_;
  tools::ThreadSafeQueue<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
    queue_{1000};

  bool open_serial();
  void configure_serial();
  void read_thread();
  void reconnect();
};

}  // namespace io

#endif  // IO__GIMBAL_HPP