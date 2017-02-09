#include "Corner_detector.h"
#include "opencv2/viz.hpp"
#include "opencv2/opencv.hpp"

void viz_test()
{
    // Create object for visualization
    cv::viz::Viz3d my_window("window 1");

    // Add world coordinate-axes
    my_window.showWidget("World-axes", cv::viz::WCoordinateSystem(1.0));

    // Add world xy-grid
    my_window.showWidget("xy-grid", cv::viz::WGrid({ 0,0,0 }, { 0,0,1 }, { 0,1,0 }, cv::Vec2i::all(20), cv::Vec2d::all(1.0),cv::viz::Color::yellow()) );

    // Render
    my_window.spin();
}


int main()
{
  //viz_test();
  cv::VideoCapture cap{1};
  if (!cap.isOpened()) return -1;

  Corner_detector det;

  std::string win_name = "Lab 3: Corner detection";
  cv::namedWindow(win_name);

  while (true)
  {
    cv::Mat frame;
    cap >> frame;

    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> corners = det.detect(gray_frame);
    cv::KeyPointsFilter::retainBest(corners, 500);

    cv::Mat viz;
    cv::drawKeypoints(frame, corners, viz, cv::Scalar{0,255,0});

    // STEG 8: Find circle from corners.

    cv::imshow(win_name, viz);
    if (cv::waitKey(30) >= 0) break;
  }
}
