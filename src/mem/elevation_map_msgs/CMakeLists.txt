cmake_minimum_required(VERSION 3.0.2)
project(elevation_map_msgs)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "/home/slsecret/opencv/build/lib")
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(OpenCV REQUIRED)

if(PYTHONLIBS_FOUND)
  message(STATUS "Using Python Libraries at: " ${PYTHON_LIBRARIES})
  message(STATUS "Using Python include directories at: " ${PYTHON_INCLUDE_DIRS})
else()
  message(WARNING "Could not find Python Libraries")
endif()

find_package(catkin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  Statistics.msg
  ChannelInfo.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  CheckSafety.srv
  Initialize.srv
)


## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
)

catkin_package(
)
