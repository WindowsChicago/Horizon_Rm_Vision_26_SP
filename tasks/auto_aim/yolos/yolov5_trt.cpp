#include "yolov5_trt.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{

YOLOV5_TRT::YOLOV5_TRT(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  model_path_ = yaml["yolov5_trt_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  // 初始化TensorRT
  logger_ = std::make_unique<Logger>();
  trt_engine_ = std::make_unique<TrtEngine>(*logger_);
  
  // 加载TensorRT引擎
  std::string engine_path = model_path_;
  size_t last_dot = engine_path.find_last_of(".");
  if (last_dot != std::string::npos) {
    engine_path = engine_path.substr(0, last_dot) + ".engine";
  }
  
  try {
    trt_engine_->LoadEngine(engine_path);
    
    // 获取输入输出维度
    auto input_dim = trt_engine_->GetInputDim();
    auto output_dim = trt_engine_->GetOutputDim();
    
    input_size_ = input_dim[2]; // 假设输入是正方形
    output_rows_ = output_dim[1]; // 25200
    output_cols_ = output_dim[2]; // 22
    
    tools::logger()->info("TensorRT engine loaded successfully. Input: {}x{}, Output: {}x{}", 
                         input_size_, input_size_, output_rows_, output_cols_);
  } catch (const std::exception& e) {
    tools::logger()->error("Failed to load TensorRT engine: {}", e.what());
    throw;
  }
}

YOLOV5_TRT::~YOLOV5_TRT()
{
  // 智能指针会自动清理
}

cv::Mat YOLOV5_TRT::preprocessImage(const cv::Mat& bgr_img, double& scale, int& pad_x, int& pad_y)
{
  // 计算缩放比例，保持宽高比
  auto x_scale = static_cast<double>(input_size_) / bgr_img.cols;
  auto y_scale = static_cast<double>(input_size_) / bgr_img.rows;
  scale = std::min(x_scale, y_scale);
  
  int new_w = static_cast<int>(bgr_img.cols * scale);
  int new_h = static_cast<int>(bgr_img.rows * scale);
  
  // 计算填充偏移（居中填充）
  pad_x = (input_size_ - new_w) / 2;
  pad_y = (input_size_ - new_h) / 2;
  
  // 调整图像大小
  cv::Mat resized_img;
  cv::resize(bgr_img, resized_img, cv::Size(new_w, new_h));
  
  // 创建填充后的图像
  cv::Mat padded_img = cv::Mat::zeros(input_size_, input_size_, CV_8UC3);
  resized_img.copyTo(padded_img(cv::Rect(pad_x, pad_y, new_w, new_h)));
  
  // 转换为RGB并归一化
  cv::Mat rgb_img;
  cv::cvtColor(padded_img, rgb_img, cv::COLOR_BGR2RGB);
  rgb_img.convertTo(rgb_img, CV_32FC3, 1.0 / 255.0);
  
  return cv::dnn::blobFromImage(rgb_img);
}

std::list<Armor> YOLOV5_TRT::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) {
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  // 预处理 - 获取缩放比例和填充偏移
  double scale;
  int pad_x, pad_y;
  cv::Mat preprocessed_img = preprocessImage(bgr_img, scale, pad_x, pad_y);

  // 分配GPU内存
  size_t input_count = 3 * input_size_ * input_size_;
  size_t output_count = output_rows_ * output_cols_;

  float *input_tensor, *output_tensor;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&input_tensor), sizeof(float) * input_count));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&output_tensor), sizeof(float) * output_count));

  // 拷贝数据到GPU
  CUDA_CHECK(cudaMemcpy(input_tensor, preprocessed_img.ptr<float>(0), 
                       sizeof(float) * input_count, cudaMemcpyHostToDevice));

  // 推理
  trt_engine_->Inference(input_tensor, output_tensor);

  // 拷贝结果回CPU
  cv::Mat output(output_rows_, output_cols_, CV_32F);
  CUDA_CHECK(cudaMemcpy(output.ptr<float>(0), output_tensor, 
                       sizeof(float) * output_count, cudaMemcpyDeviceToHost));

  // 清理GPU内存
  CUDA_CHECK(cudaFree(input_tensor));
  CUDA_CHECK(cudaFree(output_tensor));

  return parseDetections(output, scale, pad_x, pad_y, raw_img, frame_count);
}

std::list<Armor> YOLOV5_TRT::parseDetections(const cv::Mat& output, double scale, int pad_x, int pad_y, const cv::Mat& bgr_img, int frame_count)
{
  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;

  for (int r = 0; r < output.rows; r++) {
    double score = output.at<float>(r, 8);
    score = sigmoid(score);

    if (score < score_threshold_) continue;

    std::vector<cv::Point2f> armor_key_points;

    // 颜色和类别独热向量
    cv::Mat color_scores = output.row(r).colRange(9, 13);     // color
    cv::Mat classes_scores = output.row(r).colRange(13, 22);  // num
    cv::Point class_id, color_id;
    int _class_id, _color_id;
    double score_color, score_num;
    cv::minMaxLoc(classes_scores, NULL, &score_num, NULL, &class_id);
    cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);
    _class_id = class_id.x;
    _color_id = color_id.x;

    // 关键点坐标 - 修正：减去填充偏移后再除以缩放比例
    armor_key_points.push_back(
      cv::Point2f((output.at<float>(r, 0) - pad_x) / scale, (output.at<float>(r, 1) - pad_y) / scale));
    armor_key_points.push_back(
      cv::Point2f((output.at<float>(r, 6) - pad_x) / scale, (output.at<float>(r, 7) - pad_y) / scale));
    armor_key_points.push_back(
      cv::Point2f((output.at<float>(r, 4) - pad_x) / scale, (output.at<float>(r, 5) - pad_y) / scale));
    armor_key_points.push_back(
      cv::Point2f((output.at<float>(r, 2) - pad_x) / scale, (output.at<float>(r, 3) - pad_y) / scale));

    float min_x = armor_key_points[0].x;
    float max_x = armor_key_points[0].x;
    float min_y = armor_key_points[0].y;
    float max_y = armor_key_points[0].y;

    for (int i = 1; i < armor_key_points.size(); i++) {
      if (armor_key_points[i].x < min_x) min_x = armor_key_points[i].x;
      if (armor_key_points[i].x > max_x) max_x = armor_key_points[i].x;
      if (armor_key_points[i].y < min_y) min_y = armor_key_points[i].y;
      if (armor_key_points[i].y > max_y) max_y = armor_key_points[i].y;
    }

    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    color_ids.emplace_back(_color_id);
    num_ids.emplace_back(_class_id);
    boxes.emplace_back(rect);
    confidences.emplace_back(score);
    armors_key_points.emplace_back(armor_key_points);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    if (use_roi_) {
      armors.emplace_back(
        color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  tmp_img_ = bgr_img;
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }

    // 使用传统方法二次矫正角点
    if (use_traditional_) detector_.detect(*it, bgr_img);

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

std::list<Armor> YOLOV5_TRT::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // 这个方法保持与接口兼容，但实际使用parseDetections
  return parseDetections(output, scale, 0, 0, bgr_img, frame_count);
}

bool YOLOV5_TRT::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  return name_ok && confidence_ok;
}

bool YOLOV5_TRT::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);

  return name_ok;
}

cv::Point2f YOLOV5_TRT::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLOV5_TRT::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
}

void YOLOV5_TRT::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

double YOLOV5_TRT::sigmoid(double x)
{
  if (x > 0)
    return 1.0 / (1.0 + exp(-x));
  else
    return exp(x) / (1.0 + exp(x));
}

}  // namespace auto_aim