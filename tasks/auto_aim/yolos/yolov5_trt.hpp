#ifndef AUTO_AIM__YOLOV5_TRT_HPP
#define AUTO_AIM__YOLOV5_TRT_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/trt_engine.h"

namespace auto_aim
{
class YOLOV5_TRT : public YOLOBase
{
public:
  YOLOV5_TRT(const std::string & config_path, bool debug);
  ~YOLOV5_TRT();

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  std::string device_, model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_, use_traditional_;

  const int class_num_ = 13;
  const float nms_threshold_ = 0.3;
  const float score_threshold_ = 0.7;
  double min_confidence_, binary_threshold_;

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  Detector detector_;

  // TensorRT相关成员
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<TrtEngine> trt_engine_;
  size_t input_size_;
  int output_rows_;
  int output_cols_;

  // 预处理和后处理相关
  cv::Mat preprocessImage(const cv::Mat& bgr_img, double& scale, int& pad_x, int& pad_y);
  std::list<Armor> parseDetections(const cv::Mat& output, double scale, int pad_x, int pad_y, const cv::Mat& bgr_img, int frame_count);

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  double sigmoid(double x);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLOV5_TRT_HPP