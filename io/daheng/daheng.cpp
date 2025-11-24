#include "daheng.hpp"

#include <libusb-1.0/libusb.h>
#include <stdexcept>
#include "tools/logger.hpp"

using namespace std::chrono_literals;

namespace io
{
DaHengCamera::DaHengCamera(double exposure_ms, double gamma, const std::string & vid_pid)
: exposure_ms_(exposure_ms),
  gamma_(gamma),
  hDevice_(nullptr),
  quit_(false),
  ok_(false),
  queue_(1),
  vid_(-1),
  pid_(-1),
  status_(GX_STATUS_SUCCESS),
  deviceIndex_(1),  // 默认使用第一个设备
  deviceSN_(nullptr)
{
  set_vid_pid(vid_pid);
  if (libusb_init(NULL)) 
    tools::logger()->warn("Unable to init libusb!");

  // 初始化大恒相机库
  status_ = GXInitLib();
  if(status_ != GX_STATUS_SUCCESS){
    tools::logger()->warn("Unable to init DaHeng camera library!");
  }

  try_open();

  // 守护线程
  daemon_thread_ = std::thread{[this] {
    while (!quit_) {
      std::this_thread::sleep_for(100ms);

      if (ok_) continue;

      if (capture_thread_.joinable()) capture_thread_.join();

      close();
      reset_usb();
      try_open();
    }
  }};
}

DaHengCamera::~DaHengCamera()
{
  quit_ = true;
  if (daemon_thread_.joinable()) daemon_thread_.join();
  if (capture_thread_.joinable()) capture_thread_.join();
  close();
  
  // 关闭大恒相机库
  if (status_ == GX_STATUS_SUCCESS) {
    GXCloseLib();
  }
  
  tools::logger()->info("DaHengCamera destructed.");
}

void DaHengCamera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data);

  img = data.img;
  timestamp = data.timestamp;
}

void DaHengCamera::open()
{
  uint32_t nDeviceNum = 0;
  
  // 枚举设备列表
  status_ = GXUpdateDeviceList(&nDeviceNum, 1000);
  if (status_ != GX_STATUS_SUCCESS || nDeviceNum == 0) {
    throw std::runtime_error("No DaHeng camera found!");
  }

  if (deviceIndex_ > nDeviceNum) {
    throw std::runtime_error("Device index exceeds available devices!");
  }

  // 打开设备 - 使用索引方式打开，与MindVision保持一致
  streamOpenParam_.accessMode = GX_ACCESS_EXCLUSIVE;
  streamOpenParam_.openMode = GX_OPEN_INDEX;
  streamOpenParam_.pszContent = "1";  // 使用第一个设备

  status_ = GXOpenDevice(&streamOpenParam_, &hDevice_);
  if (status_ != GX_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to open DaHeng camera!");
  }

  // 获取相机能力，设置分辨率
  if (!set_resolution()) {
    throw std::runtime_error("Failed to set resolution!");
  }

  // 设置曝光时间
  if (!set_exposure_time(exposure_ms_)) {
    tools::logger()->warn("Failed to set exposure time!");
  }

  // 设置Gamma
  if (!set_gamma(gamma_)) {
    tools::logger()->warn("Failed to set gamma!");
  }

  // // 自动白平衡(1：自动;0：手动) 
  // status_ = GXSetEnum(hDevice_, GX_ENUM_BALANCE_WHITE_AUTO, 1);

  // 默认开启自动白平衡
  if (!set_white_balance_auto(true)) {
    tools::logger()->warn("Failed to set auto white balance!");
  }

  // // 手动白平衡
  // if (!set_white_balance(10,1,2)) {
  //   tools::logger()->warn("Failed to set auto white balance!");
  // }

  // 开启视频流
  if (!stream_on()) {
    throw std::runtime_error("Failed to start stream!");
  }

  // 取图线程
  capture_thread_ = std::thread{[this] {
    ok_ = true;
    while (!quit_) {
      std::this_thread::sleep_for(1ms);

      // 调用GXDQBuf取一帧图像
      status_ = GXDQBuf(hDevice_, &pFrameBuffer_, 1000);
      auto timestamp = std::chrono::steady_clock::now();

      if (status_ != GX_STATUS_SUCCESS || pFrameBuffer_->nStatus != GX_FRAME_STATUS_SUCCESS) {
        tools::logger()->warn("DaHeng camera dropped!");
        ok_ = false;
        break;
      }

      cv::Mat img;
      int64_t g_i64ColorFilter = GX_COLOR_FILTER_NONE;
      GXGetEnum(hDevice_, GX_ENUM_PIXEL_COLOR_FILTER, &g_i64ColorFilter);
      
      uint8_t *pRGB24Buf = new uint8_t[pFrameBuffer_->nWidth * pFrameBuffer_->nHeight * 3];
      uint8_t *pRaw8Image = new uint8_t[pFrameBuffer_->nWidth * pFrameBuffer_->nHeight];

      // 图像格式转换
      switch (pFrameBuffer_->nPixelFormat) {
        case GX_PIXEL_FORMAT_BAYER_GR8:
        case GX_PIXEL_FORMAT_BAYER_RG8:
        case GX_PIXEL_FORMAT_BAYER_GB8:
        case GX_PIXEL_FORMAT_BAYER_BG8: {
          DxRaw8toRGB24((unsigned char*)pFrameBuffer_->pImgBuf, pRGB24Buf, 
                       pFrameBuffer_->nWidth, pFrameBuffer_->nHeight,
                       RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(g_i64ColorFilter), false);
          break;
        }
        case GX_PIXEL_FORMAT_BAYER_GR10:
        case GX_PIXEL_FORMAT_BAYER_RG10:
        case GX_PIXEL_FORMAT_BAYER_GB10:
        case GX_PIXEL_FORMAT_BAYER_BG10:
        case GX_PIXEL_FORMAT_BAYER_GR12:
        case GX_PIXEL_FORMAT_BAYER_RG12:
        case GX_PIXEL_FORMAT_BAYER_GB12:
        case GX_PIXEL_FORMAT_BAYER_BG12: {
          DxRaw16toRaw8((unsigned char*)pFrameBuffer_->pImgBuf, pRaw8Image, 
                       pFrameBuffer_->nWidth, pFrameBuffer_->nHeight, DX_BIT_2_9);
          DxRaw8toRGB24((unsigned char*)pRaw8Image, pRGB24Buf, 
                       pFrameBuffer_->nWidth, pFrameBuffer_->nHeight,
                       RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(g_i64ColorFilter), false);
          break;
        }
        default:
          tools::logger()->warn("Unsupported pixel format!");
          break;
      }

      cv::Mat tempMat(cv::Size(pFrameBuffer_->nWidth, pFrameBuffer_->nHeight), CV_8UC3, pRGB24Buf);
      cv::cvtColor(tempMat, img, cv::COLOR_RGB2BGR);

      // 将图像buf放回库中继续采图
      status_ = GXQBuf(hDevice_, pFrameBuffer_);

      delete [] pRGB24Buf;
      delete [] pRaw8Image;

      if (!img.empty()) {
        queue_.push({img.clone(), timestamp});
      }
    }
  }};

  tools::logger()->info("DaHeng camera opened.");
}

void DaHengCamera::try_open()
{
  try {
    open();
  } catch (const std::exception & e) {
    tools::logger()->warn("{}", e.what());
  }
}

void DaHengCamera::close()
{
  if (hDevice_ == nullptr) return;
  
  status_ = GXStreamOff(hDevice_);
  status_ = GXCloseDevice(hDevice_);
  hDevice_ = nullptr;
}

bool DaHengCamera::set_exposure_time(double exposure_ms)
{
  // 大恒相机曝光时间单位是微秒
  double exposure_us = exposure_ms * 1000.0;
  status_ = GXSetFloat(hDevice_, GX_FLOAT_EXPOSURE_TIME, exposure_us);
  
  if (status_ == GX_STATUS_SUCCESS) {
    tools::logger()->info("Exposure time set to {} ms", exposure_ms);
    return true;
  } else {
    tools::logger()->warn("Failed to set exposure time!");
    return false;
  }
}

bool DaHengCamera::set_gamma(double gamma)
{
    //GX_STATUS status = GX_STATUS_SUCCESS;
    //Enables Gamma.
    status_ = GXSetBool(hDevice_, GX_BOOL_GAMMA_ENABLE, true);
    //Sets Gamma mode to user-defined mode.
    GX_GAMMA_MODE_ENTRY nValue;
    nValue = GX_GAMMA_SELECTOR_SRGB;
    status_ = GXSetEnum(hDevice_, GX_ENUM_GAMMA_MODE, nValue);
    //Gets the Gamma parameter value.
    double dColorParam = gamma;
    status_ = GXSetFloat(hDevice_, GX_FLOAT_GAMMA_PARAM, dColorParam);

    if(status_){
        std::cout << "Gamma setted success !" << std::endl;
        return true;
    }else{
        std::cout << "Gamma setted failed !" << std::endl;
        return false;
    }
}

bool DaHengCamera::set_resolution()
{
  // 这里可以根据需要设置分辨率，默认使用最大分辨率
  // 大恒相机通常不需要额外设置分辨率，使用传感器原生分辨率
  int64_t width = 0, height = 0;
  
  status_ = GXGetInt(hDevice_, GX_INT_WIDTH, &width);
  if (status_ != GX_STATUS_SUCCESS) return false;
  
  status_ = GXGetInt(hDevice_, GX_INT_HEIGHT, &height);
  if (status_ != GX_STATUS_SUCCESS) return false;
  
  width_ = static_cast<int>(width);
  height_ = static_cast<int>(height);
  
  tools::logger()->info("Camera resolution: {}x{}", width_, height_);
  return true;
}
bool DaHengCamera::set_white_balance_auto(bool enable)
{
    if (hDevice_ == nullptr) return false;
    
    status_ = GXSetEnum(hDevice_, GX_ENUM_BALANCE_WHITE_AUTO, enable ? 1 : 0);
    if (status_ == GX_STATUS_SUCCESS) {
        auto_white_balance_ = enable;
        tools::logger()->info("Auto white balance {}", enable ? "enabled" : "disabled");
        return true;
    } else {
        tools::logger()->warn("Failed to set auto white balance!");
        return false;
    }
}

bool DaHengCamera::set_white_balance(double red_gain, double green_gain, double blue_gain)
{
    if (hDevice_ == nullptr) return false;
    
    // 先关闭自动白平衡
    if (auto_white_balance_) {
        if (!set_white_balance_auto(false)) {
            return false;
        }
    }
    
    // 设置各通道增益
    status_ = GXSetEnum(hDevice_, GX_ENUM_BALANCE_RATIO_SELECTOR, GX_BALANCE_RATIO_SELECTOR_RED);
    if (status_ != GX_STATUS_SUCCESS) return false;
    status_ = GXSetFloat(hDevice_, GX_FLOAT_BALANCE_RATIO, red_gain / 10.0);
    if (status_ != GX_STATUS_SUCCESS) return false;
    
    status_ = GXSetEnum(hDevice_, GX_ENUM_BALANCE_RATIO_SELECTOR, GX_BALANCE_RATIO_SELECTOR_GREEN);
    if (status_ != GX_STATUS_SUCCESS) return false;
    status_ = GXSetFloat(hDevice_, GX_FLOAT_BALANCE_RATIO, green_gain / 10.0);
    if (status_ != GX_STATUS_SUCCESS) return false;
    
    status_ = GXSetEnum(hDevice_, GX_ENUM_BALANCE_RATIO_SELECTOR, GX_BALANCE_RATIO_SELECTOR_BLUE);
    if (status_ != GX_STATUS_SUCCESS) return false;
    status_ = GXSetFloat(hDevice_, GX_FLOAT_BALANCE_RATIO, blue_gain / 10.0);
    
    if (status_ == GX_STATUS_SUCCESS) {
        tools::logger()->info("White balance set to R:{}, G:{}, B:{}", red_gain, green_gain, blue_gain);
        return true;
    } else {
        tools::logger()->warn("Failed to set white balance!");
        return false;
    }
}



bool DaHengCamera::stream_on()
{
  status_ = GXStreamOn(hDevice_);
  if (status_ == GX_STATUS_SUCCESS) {
    tools::logger()->info("DaHeng camera stream started.");
    return true;
  } else {
    tools::logger()->warn("Failed to start camera stream!");
    return false;
  }
}

void DaHengCamera::set_vid_pid(const std::string & vid_pid)
{
  auto index = vid_pid.find(':');
  if (index == std::string::npos) {
    // 如果不是vid:pid格式，尝试解析为设备索引
    try {
      deviceIndex_ = std::stoi(vid_pid);
      tools::logger()->info("Using device index: {}", deviceIndex_);
    } catch (const std::exception &) {
      tools::logger()->warn("Invalid vid_pid: \"{}\", using default device index 1", vid_pid);
      deviceIndex_ = 1;
    }
    return;
  }

  auto vid_str = vid_pid.substr(0, index);
  auto pid_str = vid_pid.substr(index + 1);

  try {
    vid_ = std::stoi(vid_str, 0, 16);
    pid_ = std::stoi(pid_str, 0, 16);
    tools::logger()->info("USB device VID: 0x{:04X}, PID: 0x{:04X}", vid_, pid_);
  } catch (const std::exception &) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
  }
}

void DaHengCamera::reset_usb() const
{
  if (vid_ == -1 || pid_ == -1) return;

  // 使用libusb重置USB设备
  auto handle = libusb_open_device_with_vid_pid(NULL, vid_, pid_);
  if (!handle) {
    tools::logger()->warn("Unable to open USB device!");
    return;
  }

  if (libusb_reset_device(handle))
    tools::logger()->warn("Unable to reset USB device!");
  else
    tools::logger()->info("USB device reset successfully");

  libusb_close(handle);
}

}  // namespace io