#ifndef IO__DAHENG_CAMERA_HPP
#define IO__DAHENG_CAMERA_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>
#include <string>

#include "GxIAPI.h"
#include "DxImageProc.h"
#include "io/camera.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{
class DaHengCamera : public CameraBase
{
public:
  DaHengCamera(double exposure_ms, double gamma, const std::string & vid_pid);
  ~DaHengCamera() override;
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;

  // 白平衡相关功能
  bool set_white_balance_auto(bool enable);
  bool set_white_balance(double red_gain, double green_gain, double blue_gain); //一次设置全部白平衡
  bool set_white_balance(int channel, double value);  //按通道设置白平衡,可以单独设置某个颜色的白平衡


private:
  struct CameraData
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
  };

  // 曝光增益通道枚举
  enum class Channel {
    BLUE,
    GREEN,
    RED,
    ALL
  };

  double exposure_ms_, gamma_;
  GX_DEV_HANDLE hDevice_;
  int height_, width_;
  bool quit_, ok_;
  std::thread capture_thread_;
  std::thread daemon_thread_;
  tools::ThreadSafeQueue<CameraData> queue_;
  int vid_, pid_;
  
  // 大恒相机特定参数
  GX_OPEN_PARAM streamOpenParam_;
  PGX_FRAME_BUFFER pFrameBuffer_;
  GX_STATUS status_;
  uint32_t deviceIndex_;
  char* deviceSN_;

  // 白平衡参数
  bool auto_white_balance_ = true;  // 默认自动白平衡

  void open();
  void try_open();
  void close();
  void set_vid_pid(const std::string & vid_pid);
  void reset_usb() const;
  bool set_exposure_time(double exposure_ms);
  bool set_gamma(double gamma);
  bool set_resolution();
  bool stream_on();

};

}  // namespace io

#endif  // IO__DAHENG_CAMERA_HPP