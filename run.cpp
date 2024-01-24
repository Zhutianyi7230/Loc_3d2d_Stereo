#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>
#include "optimizer.h"
#include "base_type.h"
#include "converter.h"

using namespace std; 

//按逗号分割string并赋值给vector<float>
vector<float> str2vf(std::istringstream& ss, std::string& token)
{
    std::vector<float> float_values;
    while (std::getline(ss, token, ',')) {
        try {
            float value = std::stof(token);
            float_values.push_back(value);
        } catch (const std::invalid_argument& e) {
            // 处理转换错误
            std::cerr << "无效的浮点数: " << token << std::endl;
        }
    }
    return float_values;
}

cv::Mat convertToMat(const std::vector<double>& fileData) {
    cv::Mat mat(4, 4, CV_64F); // 创建一个(4, 4)的双精度浮点数矩阵

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mat.at<double>(i, j) = fileData[i * 4 + j]; // 复制数据到Mat对象
        }
    }
    return mat;
}

void LoadDataLeft(vector<double>& global2cam, vector<double>& cam2img, vector<double>& lidar2cam, vector<double>& global2ego_ini , vector<double>& global2ego, string data_dir)
{
    // string global2cam_filename = data_dir + "/global2cam_ring_rear_left.bin";
    string global2cam_filename = "/media/zhutianyi/KESU/datasets/av2_vis/02678d04-cc9f-3148-9f95-1ba66347dff9/69/global2cam_ring_rear_left_noise.bin";
    string cam2img_filename = data_dir + "/cam_intrinsic_ring_rear_left.bin";
    string lidar2cam_filename = data_dir + "/lidar2cam_ring_rear_left.bin";
    // string global2ego_filename = data_dir + "/global2ego.bin";
    string global2ego_ini_filename = "/media/zhutianyi/KESU/datasets/av2_vis/02678d04-cc9f-3148-9f95-1ba66347dff9/69/global2ego_noise_lon_lat.bin";
    string global2ego_filename = "/media/zhutianyi/KESU/datasets/av2_vis/02678d04-cc9f-3148-9f95-1ba66347dff9/69/global2ego.bin";
    
    //load global2cam
    std::ifstream inputFile(global2cam_filename, std::ios::binary);
    inputFile.read(reinterpret_cast<char*>(&global2cam[0]), 16 * sizeof(double));
    if (inputFile.gcount() != 16 * sizeof(double)) {
        std::cerr << "Error reading data from file: " << global2cam_filename << std::endl;
    }
    inputFile.close();

    //load cam2img
    std::ifstream inputFile2(cam2img_filename, std::ios::binary);
    inputFile2.read(reinterpret_cast<char*>(&cam2img[0]), 16 * sizeof(double));
    if (inputFile2.gcount() != 16 * sizeof(double)) {
        std::cerr << "Error reading data from file: " << cam2img_filename << std::endl;
    }
    inputFile2.close();

    //load lidar2cam
    std::ifstream inputFile3(lidar2cam_filename, std::ios::binary);
    inputFile3.read(reinterpret_cast<char*>(&lidar2cam[0]), 16 * sizeof(double));
    if (inputFile3.gcount() != 16 * sizeof(double)) {
        std::cerr << "Error reading data from file: " << lidar2cam_filename << std::endl;
    }
    inputFile3.close();

    //load global2ego
    std::ifstream inputFile4(global2ego_filename, std::ios::binary);
    inputFile4.read(reinterpret_cast<char*>(&global2ego[0]), 16 * sizeof(double));
    if (inputFile4.gcount() != 16 * sizeof(double)) {
        std::cerr << "Error reading data from file: " << global2ego_filename << std::endl;
    }
    inputFile4.close();

    //load global2ego_ini
    std::ifstream inputFile5(global2ego_ini_filename, std::ios::binary);
    inputFile5.read(reinterpret_cast<char*>(&global2ego_ini[0]), 16 * sizeof(double));
    if (inputFile5.gcount() != 16 * sizeof(double)) {
        std::cerr << "Error reading data from file: " << global2ego_ini_filename << std::endl;
    }
    inputFile5.close();

}

void LoadDataRight(vector<double>& global2cam, vector<double>& cam2img, vector<double>& lidar2cam, vector<double>& global2ego, string data_dir)
{
    // string global2cam_filename = data_dir + "/global2cam_ring_rear_left.bin";
    string global2cam_filename = "/media/zhutianyi/KESU/datasets/av2_vis/02678d04-cc9f-3148-9f95-1ba66347dff9/69/global2cam_ring_rear_right.bin";
    string cam2img_filename = data_dir + "/cam_intrinsic_ring_rear_right.bin";
    string lidar2cam_filename = data_dir + "/lidar2cam_ring_rear_right.bin";
    // string global2ego_filename = data_dir + "/global2ego.bin";
    string global2ego_filename = "/media/zhutianyi/KESU/datasets/av2_vis/02678d04-cc9f-3148-9f95-1ba66347dff9/69/global2ego.bin";

    //load global2cam
    std::ifstream inputFile(global2cam_filename, std::ios::binary);
    inputFile.read(reinterpret_cast<char*>(&global2cam[0]), 16 * sizeof(double));
    if (inputFile.gcount() != 16 * sizeof(double)) {
        std::cerr << "Error reading data from file: " << global2cam_filename << std::endl;
    }
    inputFile.close();

    //load cam2img
    std::ifstream inputFile2(cam2img_filename, std::ios::binary);
    inputFile2.read(reinterpret_cast<char*>(&cam2img[0]), 16 * sizeof(double));
    if (inputFile2.gcount() != 16 * sizeof(double)) {
        std::cerr << "Error reading data from file: " << cam2img_filename << std::endl;
    }
    inputFile2.close();

    //load lidar2cam
    std::ifstream inputFile3(lidar2cam_filename, std::ios::binary);
    inputFile3.read(reinterpret_cast<char*>(&lidar2cam[0]), 16 * sizeof(double));
    if (inputFile3.gcount() != 16 * sizeof(double)) {
        std::cerr << "Error reading data from file: " << lidar2cam_filename << std::endl;
    }
    inputFile3.close();

    //load global2ego
    std::ifstream inputFile4(global2ego_filename, std::ios::binary);
    inputFile4.read(reinterpret_cast<char*>(&global2ego[0]), 16 * sizeof(double));
    if (inputFile4.gcount() != 16 * sizeof(double)) {
        std::cerr << "Error reading data from file: " << global2ego_filename << std::endl;
    }
    inputFile4.close();

}

std::vector<Point3D> readPointData(const std::string &filename) {
    std::vector<Point3D> pointData;
    std::ifstream inputFile(filename);

    if (!inputFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return pointData;
    }
    int currentIndex = 0;  // 当前点的索引
    std::string line;
    while (std::getline(inputFile, line)) {
        //是tensor行
        if (line.find("tensor") == 0) {
            continue;
        } 
        else if (!line.empty()) {
            // 解析坐标数据
            std::istringstream iss(line);
            vector<float> float_values = str2vf(iss, line);//
            float x, y, z;
            x = float_values[0];
            y = float_values[1];
            z = float_values[2];
            Point3D point(x,y,z);
            point.index = currentIndex;
            pointData.push_back(point);
            currentIndex++;
        }
    }
    inputFile.close();
    return pointData;
}

std::vector<Line> readLineData(const std::string &filename)
{
    vector<Line> lineData;
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return lineData;
    }
    int currentIndex = 0;  // 当前点的索引
    std::string line;
    Line currentLine({0, 0}, {0, 0});

    while (std::getline(inputFile, line)) {
        bool isBlankLine = std::all_of(line.begin(), line.end(), [](char c) { return std::isspace(c); });//判断是否空行
        if (!isBlankLine) {
            // 解析坐标数据
            std::istringstream iss(line);
            vector<float> float_values = str2vf(iss, line);
            float x, y;
            x = float_values[0];
            y = float_values[1];
            Point2D point(x, y);
            if(currentIndex != 0)
            {
                currentLine.points[1] = point;
                currentLine.index = currentIndex;
                lineData.push_back(currentLine);
            }
            currentLine.points[0] = point;
            currentIndex++;
        }
        else{
            currentIndex = 0;
            Line currentLine({0, 0}, {0, 0});
        }
    }
    inputFile.close();
    return lineData;

}

int main( int argc, char** argv )
{
    vector<double> global2caml(16);
    vector<double> cam2imgl(16);
    vector<double> lidar2caml(16);

    vector<double> global2camr(16);
    vector<double> cam2imgr(16);
    vector<double> lidar2camr(16);
    vector<double> global2ego_ini(16);
    vector<double> global2ego(16);
    int nGood;
    std::string data_dir = "/media/zhutianyi/KESU/datasets/av2_vis/02678d04-cc9f-3148-9f95-1ba66347dff9/69";
    //读取地图点坐标
    // string pts_filename_l = data_dir + "/69_ring_rear_left.txt";
    string pts_filename_l = "/media/zhutianyi/KESU/datasets/av2_vis/scene1pts/69/69_ring_rear_left.txt";
    vector<Point3D> pointset_l = readPointData(pts_filename_l);
    string pts_filename_r = "/media/zhutianyi/KESU/datasets/av2_vis/scene1pts/69/69_ring_rear_right.txt";
    vector<Point3D> pointset_r = readPointData(pts_filename_r);
    //读取线坐标
    string line_filename_l = "/media/zhutianyi/KESU/datasets/av2_vis/scene1_ring_rear_left/line2d/69.txt";
    vector<Line> lineset_l = readLineData(line_filename_l);
    string line_filename_r = "/media/zhutianyi/KESU/datasets/av2_vis/scene1_ring_rear_right/line2d/69.txt";
    vector<Line> lineset_r = readLineData(line_filename_r);

    cout << "pointset.size = " << pointset_l.size()+pointset_r.size() << endl;
    cout << "lineset.size = " << lineset_l.size()+lineset_r.size() << endl;

    //先从txt读入数据
    LoadDataLeft(global2caml, cam2imgl, lidar2caml, global2ego_ini, global2ego, data_dir);
    LoadDataRight(global2camr, cam2imgr, lidar2camr, global2ego, data_dir);
    // //把vector转换成cv::Mat
    cv::Mat instrincsl = convertToMat(cam2imgl);
    cv::Mat instrincsr = convertToMat(cam2imgr);
    cv::Mat extrincsl = convertToMat(lidar2caml);
    cv::Mat extrincsr = convertToMat(lidar2camr);
    cv::Mat mglobal2ego = convertToMat(global2ego);
    cv::Mat mglobal2ego_ini = convertToMat(global2ego_ini);

    // cout << mglobal2ego_ini << endl;
    cv::Mat mego2global_ini = cv::Mat_<double>(4,4);
    bool invertible = cv::invert(mglobal2ego_ini, mego2global_ini);
    cout << "Ini e2g:" << mego2global_ini.at<double>(0,3) << " " << mego2global_ini.at<double>(1,3) << " " << mego2global_ini.at<double>(2,3) << endl;

    cv::Mat mego2global = cv::Mat_<double>(4,4);
    bool invertible2 = cv::invert(mglobal2ego, mego2global);
    cout << "GT e2g:" << mego2global.at<double>(0,3) << " " << mego2global.at<double>(1,3) << " " << mego2global.at<double>(2,3) << endl;


    //初始化optimizer实例
    Optimizer optimizer(mglobal2ego_ini, instrincsl, instrincsr, extrincsl, extrincsr);

    optimizer.SearchPairLeft(pointset_l, lineset_l);//寻找最近的line，并关联最近点（垂足）
    optimizer.SearchPairRight(pointset_r, lineset_r);

    cv::Mat pose = optimizer.PoseOptimization(pointset_l, pointset_r, lineset_l, lineset_r);//图优化
    cv::Mat pose_T = cv::Mat_<double>(4,4);
    bool inv = cv::invert(pose, pose_T);

    cout << "Pred e2g:" << pose_T.at<double>(0,3) << " " << pose_T.at<double>(1,3) << " " << pose_T.at<double>(2,3) << endl;

    //计算纵向/横向误差
    Vector4d ini_b_world(mego2global_ini.at<double>(0,3), mego2global_ini.at<double>(1,3), mego2global_ini.at<double>(2,3), mego2global_ini.at<double>(3,3));
    Eigen::Matrix<double,4,4> eigen_global2ego_gt;
    eigen_global2ego_gt = Converter::toMatrix4d(mglobal2ego);
    Vector4d b_ego_ini = eigen_global2ego_gt * ini_b_world;
    cout << "Ini Loc error:" << b_ego_ini[0] << " " << b_ego_ini[1] << " " << b_ego_ini[2] << endl;

    Vector4d b_world(pose_T.at<double>(0,3), pose_T.at<double>(1,3), pose_T.at<double>(2,3), pose_T.at<double>(3,3));
    Vector4d b_ego_pred = eigen_global2ego_gt * b_world;
    cout << "Loc error:" << b_ego_pred[0] << " " << b_ego_pred[1] << " " << b_ego_pred[2] << endl;

    cout << "Finish PoseOptimization." << endl;

    return 0;
}

//GT e2g:6736.3 1659.27 60.2908