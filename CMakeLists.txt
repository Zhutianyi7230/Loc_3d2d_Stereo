cmake_minimum_required( VERSION 2.8 )
project( loc3d2d )

set( CMAKE_BUILD_TYPE "Release" )
# set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(CMAKE_CXX_FLAGS "-std=c++17 -O2 ${SSE_FLAGS} -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# 寻找G2O
find_package( Eigen3 REQUIRED )
find_package( G2O REQUIRED )
include_directories( 
    include
    ${G2O_INCLUDE_DIRS}
    "/usr/include/eigen3"
)

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )


file(GLOB SOURCES "src/*.cpp")
add_library( optimizer SHARED ${SOURCES}) 
target_link_libraries(optimizer ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} g2o_types_sba)

add_executable( loc3d2d run.cpp )
# 与G2O和OpenCV链接
target_link_libraries( loc3d2d 
    ${OpenCV_LIBS}
    optimizer
    g2o_core g2o_stuff
)
