#include "Corner_detector.h"

using namespace std;

std::vector<cv::KeyPoint> Corner_detector::detect(cv::Mat image) const
{
  cv::Mat Ix;
  cv::Mat Iy;

  // STEP 2: Estimate image gradients Ix and Iy using g_kernel_ and dg_kernel.
  cv::sepFilter2D(image, Ix, CV_32F, dg_kernel_, g_kernel_);
  cv::sepFilter2D(image, Iy, CV_32F, g_kernel_, dg_kernel_);

  // STEP 3: Compute the elements of M; A, B and C from Ix and Iy.
  cv::Mat A;
  cv::Mat B;
  cv::Mat C;

  A = Ix.mul(Ix);
  B = Ix.mul(Iy);
  C = Iy.mul(Iy);

  // STEP 3: Apply the windowing gaussian win_kernel on A, B and C.
  cv::sepFilter2D(A, A, -1, win_kernel, win_kernel);
  cv::sepFilter2D(B, B, -1, win_kernel, win_kernel);
  cv::sepFilter2D(B, C, -1, win_kernel, win_kernel);

  // STEP 4: Finish all the corner response functions.
  cv::Mat response;
  switch (metric_type_)
  {
  case Corner_metric::harris:
    response = harris_metric(A, B, C); break;

  case Corner_metric::harmonic_mean:
    response = harmonic_mean_metric(A, B, C); break;

  case Corner_metric::min_eigen:
    response = min_eigen_metric(A, B, C); break;
  }

  // STEP 5: Compute the threshold by using quality_level_ on the maximum response.
  double maxval{0};
  cv::minMaxLoc(response, nullptr, &maxval);

  // STEP 5: Threshold the response.
  cv::threshold(response, response, maxval * quality_level_, 0, cv::THRESH_TOZERO);

  // STEP 6: Find the local maxima, and extract corners.
  cv::Mat localMax;
  cv::dilate(response, localMax, cv::Mat{});

  cv::Size img_size = image.size();
  std::vector<cv::KeyPoint> key_points;
  float keypoint_size = static_cast<float>(3.0 * window_sigma_);
  for (int y = 1; y < img_size.height - 1; ++y) {
    for (int x = 1; x < img_size.width - 1; ++x) {
      float val = response.at<float>(y, x);
      float local_max_val = localMax.at<float>(y, x);

      if (val != 0 && val == local_max_val) {
        cv::Point2f point(static_cast<float>(x), static_cast<float>(y));
        key_points.push_back(cv::KeyPoint{point, keypoint_size, -1, val});
      }
    }
  }
  return key_points;
}

// STEP 4: Finish function
cv::Mat Corner_detector::harris_metric(cv::Mat& A, cv::Mat& B, cv::Mat& C) const
{
  cv::Mat harris;
  cv::Mat detM = A.mul(C) - B.mul(B);
  cv::Mat traceM = A + C;
  harris = detM - 0.06 * traceM.mul(traceM);
  return harris;
}

// STEP 4: Finish function
cv::Mat Corner_detector::harmonic_mean_metric(cv::Mat& A, cv::Mat& B, cv::Mat& C) const
{
  cv::Mat harmonic;
  cv::Mat detM = A.mul(C) - B.mul(B);
  cv::Mat traceM = A + C;
  return detM.mul(1.0 / traceM);
}

// STEP 4: Finish function
cv::Mat Corner_detector::min_eigen_metric(cv::Mat& A, cv::Mat& B, cv::Mat& C) const
{
  cv::Mat root;
  cv::sqrt(4 * B.mul(B) + (A - C).mul(A - C), root);
  return 0.5*((A + C) - root);
}

cv::Mat gaussian_kernel(double sigma, int radius)
{
  if (radius <= 0) radius = std::ceil(3.5 * sigma);

  int length = 2*radius + 1;
  
  cv::Mat kernel{length, 1, CV_64F};
  double* element = kernel.ptr<double>();

  double scale{-0.5 / (sigma*sigma)};
  double sum{0};
  for (int i = 0; i < length; ++i)
  {
    int x = i - radius;
    element[i] = std::exp(x*x*scale);
    sum += element[i];
  }

  kernel /= sum;
  return kernel;
}

// STEP 1: Finish function
cv::Mat derivated_gaussian_kernel(double sigma, int radius)
{
  cv::Mat gauss = gaussian_kernel(sigma, radius);
  if (radius <= 0) radius = std::ceil(3.5 * sigma);

  int length = 2*radius + 1;

  double* element = gauss.ptr<double>();

  double sigma_sqr = sigma * sigma;
  for (int i = 0; i < length; ++i)
  {
    int x = i - radius;
    element[i] = -1 * x * element[i] / sigma_sqr;
  }
  return gauss;
}
