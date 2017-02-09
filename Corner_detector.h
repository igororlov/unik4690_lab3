#ifndef LAB_3_CORNER_DETECTOR_H
#define LAB_3_CORNER_DETECTOR_H

#include "opencv2/opencv.hpp"

cv::Mat gaussian_kernel(double sigma, int radius = 0);
cv::Mat derivated_gaussian_kernel(double sigma, int radius = 0);

enum class Corner_metric {harris, harmonic_mean, min_eigen};

class Corner_detector
{
public:
  Corner_detector(
    Corner_metric metric = Corner_metric::harris,
    double gradient_sigma = 1.0,
    double window_sigma = 2.0, 
    double quality_level = 0.01)
    : metric_type_{metric},
    window_sigma_{window_sigma},
    quality_level_{quality_level},
    g_kernel_{gaussian_kernel(gradient_sigma)},
    dg_kernel_{derivated_gaussian_kernel(gradient_sigma)},
    win_kernel{gaussian_kernel(window_sigma_)} {}

  std::vector<cv::KeyPoint> detect(cv::Mat image) const;

private:
  cv::Mat harris_metric(cv::Mat& A, cv::Mat& B, cv::Mat& C) const;
  cv::Mat harmonic_mean_metric(cv::Mat& A, cv::Mat& B, cv::Mat& C) const;
  cv::Mat min_eigen_metric(cv::Mat& A, cv::Mat& B, cv::Mat& C) const;

  Corner_metric metric_type_;
  double window_sigma_;
  double quality_level_;
  cv::Mat g_kernel_;
  cv::Mat dg_kernel_;
  cv::Mat win_kernel;
};

#endif