cmake_minimum_required(VERSION 3.8)
project(rl_infer_cpp)

# Latency matters (100 Hz control loop) — default to an optimized build.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_index_cpp REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIRS})

# ---------- hover node ----------
add_executable(hover_jax_node
  src/base_node.cpp
  src/hover_node.cpp
)
target_compile_features(hover_jax_node PUBLIC cxx_std_17)
ament_target_dependencies(hover_jax_node
  rclcpp px4_msgs std_msgs geometry_msgs nav_msgs ament_index_cpp)

# ---------- gate node ----------
add_executable(gate_jax_node
  src/base_node.cpp
  src/gate_node.cpp
)
target_compile_features(gate_jax_node PUBLIC cxx_std_17)
ament_target_dependencies(gate_jax_node
  rclcpp px4_msgs std_msgs geometry_msgs nav_msgs tf2_msgs ament_index_cpp)

# ---------- offline validator / benchmark (no ROS deps) ----------
add_executable(rl_validate src/validate_main.cpp)
target_compile_features(rl_validate PUBLIC cxx_std_17)

install(TARGETS hover_jax_node gate_jax_node rl_validate
  DESTINATION lib/${PROJECT_NAME})

ament_package()
