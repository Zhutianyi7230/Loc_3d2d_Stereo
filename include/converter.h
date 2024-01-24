#ifndef CONVERTER_H
#define CONVERTER_H

#include<opencv2/core/core.hpp>
#include<Eigen/Dense>
#include <g2o/types/sim3/types_seven_dof_expmap.h>


class Converter
{
public:
    static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);

    static cv::Mat toCvMat(const g2o::SE3Quat &SE3);

    static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);

    static Eigen::Matrix<double,4,4> toMatrix4d(const cv::Mat &cvMat4);

};

#endif