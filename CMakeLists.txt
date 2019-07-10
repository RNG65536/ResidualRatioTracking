cmake_minimum_required(VERSION 3.12)
project(ratio_tracking)

find_package(GLUT REQUIRED)
find_package(GLM REQUIRED)

add_executable(ratio_tracking main.cpp)

target_include_directories(ratio_tracking PRIVATE ${GLUT_INCLUDE_DIR} ${GLM_INCLUDE_DIRS})
target_link_libraries(ratio_tracking PRIVATE debug ${GLUT_glut_LIBRARY_DEBUG} optimized ${GLUT_glut_LIBRARY_RELEASE})