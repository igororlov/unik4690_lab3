cmake_minimum_required( VERSION 2.8 )
project( Lab_3 )

# Add libraries.
find_package(OpenCV 3 REQUIRED)

add_executable( Lab_3 
  Corner_detector.h
  Corner_detector.cpp
  main.cpp
)

target_link_libraries( Lab_3 
  ${OpenCV_LIBS}
)
