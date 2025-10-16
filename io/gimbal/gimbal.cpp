#include "gimbal.hpp"

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <iomanip>

namespace io
{
Gimbal::Gimbal(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  com_port_ = tools::read<std::string>(yaml, "com_port");

  tools::logger()->info("[Gimbal] Initializing gimbal communication on port: {}", com_port_);

  if (!open_serial()) {
    tools::logger()->error("[Gimbal] Failed to open serial port: {}", com_port_);
    exit(1);
  }

  thread_ = std::thread(&Gimbal::read_thread, this);

  // Wait for first data
  queue_.pop();
  tools::logger()->info("[Gimbal] First quaternion data received.");
}

Gimbal::~Gimbal()
{
  tools::logger()->info("[Gimbal] Shutting down gimbal communication");
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  if (fd_ >= 0) {
    close(fd_);
    tools::logger()->info("[Gimbal] Serial port closed");
  }
}

bool Gimbal::open_serial()
{
  tools::logger()->info("[Gimbal] Opening serial port: {}", com_port_);
  fd_ = open(com_port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) {
    tools::logger()->error("[Gimbal] Failed to open serial port {}: {}", com_port_, strerror(errno));
    return false;
  }
  
  tools::logger()->info("[Gimbal] Successfully opened serial port, fd: {}", fd_);
  configure_serial();
  return true;
}

void Gimbal::configure_serial()
{
  tools::logger()->info("[Gimbal] Configuring serial port parameters");
  struct termios options;
  tcgetattr(fd_, &options);
  
  // Set baud rate
  cfsetispeed(&options, B115200);
  cfsetospeed(&options, B115200);
  
  // Enable receiver and set local mode
  options.c_cflag |= (CLOCAL | CREAD);
  
  // Set data bits, stop bits, parity
  options.c_cflag &= ~PARENB;  // No parity
  options.c_cflag &= ~CSTOPB;  // 1 stop bit
  options.c_cflag &= ~CSIZE;
  options.c_cflag |= CS8;      // 8 data bits
  
  // Disable software flow control
  options.c_iflag &= ~(IXON | IXOFF | IXANY);
  options.c_iflag &= ~(INLCR | IGNCR | ICRNL);
  
  // Raw input
  options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
  options.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP);
  
  // Raw output
  options.c_oflag &= ~OPOST;
  
  // Set timeouts - return immediately with available data
  options.c_cc[VMIN] = 0;
  options.c_cc[VTIME] = 0;
  
  tcsetattr(fd_, TCSANOW, &options);
  tcflush(fd_, TCIOFLUSH);
  
  tools::logger()->info("[Gimbal] Serial port configured: 115200 baud, 8N1, no flow control");
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  while (true) {
    auto [q_a, t_a] = queue_.pop();
    auto [q_b, t_b] = queue_.front();
    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t);
    auto k = t_ac / t_ab;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
    if (t < t_a) return q_c;
    if (!(t_a < t && t <= t_b)) continue;

    return q_c;
  }
}

std::string Gimbal::packet_to_hex(const void* data, size_t size) const
{
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  std::stringstream ss;
  ss << std::hex << std::setfill('0');
  for (size_t i = 0; i < size; ++i) {
    ss << std::setw(2) << static_cast<int>(bytes[i]);
    if (i < size - 1) ss << " ";
  }
  return ss.str();
}

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  // 复制数据到局部变量以避免packed结构体引用问题
  uint8_t mode = VisionToGimbal.mode;
  float yaw = VisionToGimbal.yaw;
  float pitch = VisionToGimbal.pitch;
  
  // 赋值给tx_data_
  tx_data_.mode = mode;
  tx_data_.yaw = yaw;
  tx_data_.pitch = pitch;
  tx_data_.timestamp = 0;  // 时间戳暂时填0
  
  if (fd_ < 0) {
    tools::logger()->error("[Gimbal] Cannot send data - serial port not open");
    return;
  }
  
  // 使用局部变量记录发送的数据内容
  tools::logger()->debug("[Gimbal] Sending data - Mode: {}, Pitch: {:.3f}, Yaw: {:.3f}",
                        mode, pitch, yaw);
  
  ssize_t bytes_written = write(fd_, &tx_data_, sizeof(tx_data_));
  if (bytes_written != sizeof(tx_data_)) {
    tools::logger()->warn("[Gimbal] Failed to write serial, expected {} bytes, got {} bytes, error: {}",
                         sizeof(tx_data_), bytes_written, strerror(errno));
  } else {
    tools::logger()->debug("[Gimbal] Successfully sent {} bytes to gimbal", bytes_written);
  }
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  //uint8_t mode = control ? (fire ? 2 : 1) : 0; //等价于下面的if else
  // uint8_t mode;
  // if (control) 
  // {
  //     if (fire) 
  //     {
  //         mode = 2;  // 控制且开火
  //     } 
  //     else 
  //     {
  //         mode = 1;  // 控制但不开火
  //     }
  // } 
  // else 
  // {
  //     mode = 0;      // 不控制
  // }

  uint8_t mode;
  if (control) 
  {
      if (fire) 
      {
          mode = 57;  // 控制且开火，对应NT_M6的111001
      } 
      else 
      {
          mode = 49;  // 控制但不开火，对应NT_M6的110001
      }
  } 
  else 
  {
      mode = 1;      // 不控制，，对应NT_M6的自瞄模式默认标志位001
  }

  // 赋值给tx_data_
  tx_data_.mode = mode;
  tx_data_.yaw = yaw;
  tx_data_.pitch = pitch;
  tx_data_.timestamp = 0;  // 时间戳暂时填0
  
  if (fd_ < 0) {
    tools::logger()->error("[Gimbal] Cannot send data - serial port not open");
    return;
  }
  
  // 使用局部变量记录发送的数据内容
  std::string mode_str = control ? (fire ? "CONTROL_FIRE" : "CONTROL_NO_FIRE") : "NO_CONTROL";
  tools::logger()->debug("[Gimbal] Sending data - Mode: {} ({}), Pitch: {:.3f}, Yaw: {:.3f}",
                        mode_str, mode, pitch, yaw);
  
  ssize_t bytes_written = write(fd_, &tx_data_, sizeof(tx_data_));
  if (bytes_written != sizeof(tx_data_)) {
    tools::logger()->warn("[Gimbal] Failed to write serial, expected {} bytes, got {} bytes, error: {}",
                         sizeof(tx_data_), bytes_written, strerror(errno));
  } else {
    tools::logger()->debug("[Gimbal] Successfully sent {} bytes to gimbal", bytes_written);
    
    // 记录原始数据
    tools::logger()->trace("[Gimbal] Raw TX data: {}", 
                          packet_to_hex(&tx_data_, sizeof(tx_data_)));
  }
}

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;
  const size_t packet_size = sizeof(GimbalToVision);
  
  uint8_t buffer[1024];
  ssize_t bytes_read;
  size_t data_index = 0;
  
  while (!quit_) {
    if (error_count > 5000) {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Too many errors ({}), attempting to reconnect...", error_count);
      reconnect();
      continue;
    }
    
    if (fd_ < 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    
    bytes_read = read(fd_, buffer + data_index, sizeof(buffer) - data_index);
    if (bytes_read < 0) {
      if (errno != EAGAIN && errno != EWOULDBLOCK) {
        error_count++;
        tools::logger()->debug("[Gimbal] Read error: {} (errno: {})", strerror(errno), errno);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    
    if (bytes_read == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    
    // 记录接收的原始数据
    tools::logger()->trace("[Gimbal] Received {} bytes raw data: {}", 
                          bytes_read, packet_to_hex(buffer + data_index, bytes_read));
    
    data_index += bytes_read;
    
    // Process complete packets
    bool packet_found = false;
    for (size_t i = 0; i <= data_index - 2; i++) {
      // Look for packet header 0xCD
      if (buffer[i] == 0xCD) {
        // Check if we have a complete packet
        if (i + packet_size <= data_index) {
          // Check packet tail
          if (buffer[i + packet_size - 1] != 0xDC) {
            tools::logger()->warn("[Gimbal] Packet tail mismatch, expected 0xDC, got 0x{:02x}", 
                                 buffer[i + packet_size - 1]);
            continue;
          }
          
          auto t = std::chrono::steady_clock::now();
          
          // Copy valid packet
          std::memcpy(&rx_data_, buffer + i, packet_size);
          
          // 记录原始数据包
          tools::logger()->debug("[Gimbal] Found complete packet at offset {}, raw: {}", 
                                i, packet_to_hex(buffer + i, packet_size));
          
          // 复制到局部变量以避免packed结构体引用问题
          uint8_t mode = rx_data_.mode;
          float yaw = rx_data_.yaw;
          float pitch = rx_data_.pitch;
          uint8_t bullet_speed = rx_data_.bullet_speed;
          
          // 使用yaw和pitch计算四元数（roll设为0）
          Eigen::AngleAxisd yaw_angle(yaw, Eigen::Vector3d::UnitZ());
          Eigen::AngleAxisd pitch_angle(pitch, Eigen::Vector3d::UnitY());
          Eigen::AngleAxisd roll_angle(0.0, Eigen::Vector3d::UnitX());
          
          Eigen::Quaterniond q = yaw_angle * pitch_angle * roll_angle;
          q.normalize();
          
          // Validate mode field to avoid invalid mode warnings
          if (mode <= 3) {
            // Process the packet
            queue_.push({q, t});
            
            std::lock_guard<std::mutex> lock(mutex_);
            state_.yaw = yaw;
            state_.yaw_vel = 0.0f;  // 速度暂时填0
            state_.pitch = pitch;
            state_.pitch_vel = 0.0f;  // 速度暂时填0
            state_.bullet_speed = static_cast<float>(bullet_speed);
            state_.bullet_count = 0;  // 子弹计数暂时填0
            
            GimbalMode old_mode = mode_;
            switch (mode) {
              case 0:
                mode_ = GimbalMode::IDLE;
                break;
              case 1:
                mode_ = GimbalMode::AUTO_AIM;
                break;
              case 2:
                mode_ = GimbalMode::SMALL_BUFF;
                break;
              case 3:
                mode_ = GimbalMode::BIG_BUFF;
                break;
              default:
                mode_ = GimbalMode::IDLE;
                break;
            }
            
            // 使用局部变量记录解析后的数据内容
            tools::logger()->info("[Gimbal] Parsed data - Mode: {}->{}, Pitch: {:.3f}, Yaw: {:.3f}, "
                                 "BulletSpeed: {}, Quaternion: [{:.3f}, {:.3f}, {:.3f}, {:.3f}]",
                                 str(old_mode), str(mode_), pitch, yaw, bullet_speed, 
                                 q.w(), q.x(), q.y(), q.z());
            
            // Move remaining data to beginning of buffer
            size_t remaining = data_index - (i + packet_size);
            if (remaining > 0) {
              std::memmove(buffer, buffer + i + packet_size, remaining);
            }
            data_index = remaining;
            packet_found = true;
            error_count = 0;
            break;
          } else {
            // Invalid mode, skip this packet but continue processing
            tools::logger()->warn("[Gimbal] Skipping packet with invalid mode: {}, raw data: {}", 
                                 mode, packet_to_hex(buffer + i, packet_size));
            // Move to next byte and continue searching
            i += 1;
          }
        } else {
          // Incomplete packet, wait for more data
          tools::logger()->trace("[Gimbal] Incomplete packet, have {} bytes, need {} bytes", 
                                data_index - i, packet_size);
          break;
        }
      }
    }
    
    // If no packet found and buffer is full, clear some data to prevent overflow
    if (!packet_found && data_index >= sizeof(buffer) - 10) {
      tools::logger()->warn("[Gimbal] Buffer overflow protection, clearing buffer. data_index: {}", data_index);
      // Keep last 100 bytes in case header is split
      if (data_index > 100) {
        std::memmove(buffer, buffer + data_index - 100, 100);
        data_index = 100;
        tools::logger()->debug("[Gimbal] Kept last 100 bytes, new data_index: {}", data_index);
      }
      error_count++;
    }
  }
  
  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  tools::logger()->info("[Gimbal] Attempting to reconnect to serial port");
  
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
    tools::logger()->info("[Gimbal] Closed existing serial connection");
  }
  
  std::this_thread::sleep_for(std::chrono::seconds(1));
  
  if (!open_serial()) {
    tools::logger()->error("[Gimbal] Reconnect failed for port: {}", com_port_);
  } else {
    tools::logger()->info("[Gimbal] Reconnected successfully to port: {}", com_port_);
    // Clear buffer and reset state
    std::lock_guard<std::mutex> lock(mutex_);
    mode_ = GimbalMode::IDLE;
    state_ = GimbalState{};
  }
}

}  // namespace io