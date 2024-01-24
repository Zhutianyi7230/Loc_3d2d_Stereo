#include "../include/optimizer.h"
#include "../include/converter.h"
#include "../include/base_type.h"
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>
#include <vector>


Vector2d Optimizer::project2d(const Vector3d& v)  {
  Vector2d res;
  res(0) = v(0)/v(2);
  res(1) = v(1)/v(2);
  return res;
}

Vector2d Optimizer::left_cam_project(const Vector3d & trans_xyz)
{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx_l + cx_l;
  res[1] = proj[1]*fy_l + cy_l;
  return res;
}

Vector2d Optimizer::right_cam_project(const Vector3d & trans_xyz)
{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx_r + cx_r;
  res[1] = proj[1]*fy_r + cy_r;
  return res;
}

void Optimizer::SearchPairLeft(vector<Point3D> &pointset, vector<Line> &lineset)
{
  cv::Mat mTgc = mego2caml * mTge;
  g2o::SE3Quat SE3quat = Converter::toSE3Quat(mTgc);
  std::vector<int> closest_line_indices(pointset.size(), -1); // 用于存储每个点对应的最近线段的索引

  for (size_t i = 0; i < pointset.size(); ++i)
  {
    Point3D &map_point = pointset[i];
    Vector3d pt3d;
    pt3d[0] = map_point.x;
    pt3d[1] = map_point.y;
    pt3d[2] = map_point.z;

    Vector2d pt2d = left_cam_project(SE3quat.map(pt3d));////////投影  3D->2D
    Point2D pt(pt2d[0],pt2d[1]);

    float min_distance = std::numeric_limits<float>::max(); // 初始化最小距离为一个大数
    int closest_line_index = -1;
    Point2D closest_point;

    for (size_t j = 0; j < lineset.size(); j++)
    {
        const Line &line = lineset[j];
        Point2D foot_point;
        const Point2D &line_start = line.points[0];
        const Point2D &line_end = line.points[1];

        // 计算点到线段的垂足
        float u = ((pt.x - line_start.x) * (line_end.x - line_start.x) + (pt.y - line_start.y) * (line_end.y - line_start.y)) /
                  ((line_end.x - line_start.x) * (line_end.x - line_start.x) + (line_end.y - line_start.y) * (line_end.y - line_start.y));
        float dist;

        if (u >= 0.0 && u <= 1.0) // 垂足在线段内
        {
            foot_point.x = line_start.x + u * (line_end.x - line_start.x);
            foot_point.y = line_start.y + u * (line_end.y - line_start.y);
            // 计算点到垂足的距离
            dist = std::sqrt((foot_point.x - pt.x) * (foot_point.x - pt.x) + (foot_point.y - pt.y) * (foot_point.y - pt.y));
        }
        else
        {
          float dist1 = std::sqrt((line_start.x - pt.x) * (line_start.x - pt.x) + (line_start.y - pt.y) * (line_start.y - pt.y));
          float dist2 = std::sqrt((line_end.x - pt.x) * (line_end.x - pt.x) + (line_end.y - pt.y) * (line_end.y - pt.y));
          if(dist1<dist2){
            dist = dist1;
            foot_point.x = line_start.x;
            foot_point.y = line_start.y;
          }
          else{
            dist = dist2;
            foot_point.x = line_end.x;
            foot_point.y = line_end.y;
          }
        }
        // 如果距离比当前最小距离小，则更新最小距离和最近线段索引
        if (dist < min_distance)
        {
            min_distance = dist;
            closest_line_index = j;
            closest_point.x = foot_point.x;
            closest_point.y = foot_point.y;
        }
    }

    closest_line_indices[i] = closest_line_index;
    map_point.npoint = closest_point;
  }
  cout << "Finish SearchPairLeft." << endl;
}

void Optimizer::SearchPairRight(vector<Point3D> &pointset, vector<Line> &lineset)
{
  cv::Mat mTgc = mego2camr * mTge;
  g2o::SE3Quat SE3quat = Converter::toSE3Quat(mTgc);
  std::vector<int> closest_line_indices(pointset.size(), -1); // 用于存储每个点对应的最近线段的索引

  for (size_t i = 0; i < pointset.size(); ++i)
  {
    Point3D &map_point = pointset[i];
    Vector3d pt3d;
    pt3d[0] = map_point.x;
    pt3d[1] = map_point.y;
    pt3d[2] = map_point.z;

    Vector2d pt2d = right_cam_project(SE3quat.map(pt3d));////////投影  3D->2D
    Point2D pt(pt2d[0],pt2d[1]);

    float min_distance = std::numeric_limits<float>::max(); // 初始化最小距离为一个大数
    int closest_line_index = -1;
    Point2D closest_point;

    for (size_t j = 0; j < lineset.size(); j++)
    {
        const Line &line = lineset[j];
        Point2D foot_point;
        const Point2D &line_start = line.points[0];
        const Point2D &line_end = line.points[1];
        // 计算点到线段的垂足
        float u = ((pt.x - line_start.x) * (line_end.x - line_start.x) + (pt.y - line_start.y) * (line_end.y - line_start.y)) /
                  ((line_end.x - line_start.x) * (line_end.x - line_start.x) + (line_end.y - line_start.y) * (line_end.y - line_start.y));
        float dist;
        if (u >= 0.0 && u <= 1.0) // 垂足在线段内
        {
            foot_point.x = line_start.x + u * (line_end.x - line_start.x);
            foot_point.y = line_start.y + u * (line_end.y - line_start.y);
            // 计算点到垂足的距离
            dist = std::sqrt((foot_point.x - pt.x) * (foot_point.x - pt.x) + (foot_point.y - pt.y) * (foot_point.y - pt.y));
        }
        else
        {
          float dist1 = std::sqrt((line_start.x - pt.x) * (line_start.x - pt.x) + (line_start.y - pt.y) * (line_start.y - pt.y));
          float dist2 = std::sqrt((line_end.x - pt.x) * (line_end.x - pt.x) + (line_end.y - pt.y) * (line_end.y - pt.y));
          if(dist1<dist2){
            dist = dist1;
            foot_point.x = line_start.x;
            foot_point.y = line_start.y;
          }
          else{
            dist = dist2;
            foot_point.x = line_end.x;
            foot_point.y = line_end.y;
          }
        }
        // 如果距离比当前最小距离小，则更新最小距离和最近线段索引
        if (dist < min_distance)
        {
            min_distance = dist;
            closest_line_index = j;
            closest_point.x = foot_point.x;
            closest_point.y = foot_point.y;
        }
    }

    closest_line_indices[i] = closest_line_index;
    map_point.npoint = closest_point;
  }

  cout << "Finish SearchPairRight." << endl;
}

cv::Mat Optimizer::PoseOptimization(vector<Point3D> &pointset_l, vector<Point3D> &pointset_r, vector<Line> &lineset_l, vector<Line> &lineset_r)
{  
  // Step 1：构造g2o优化器, BlockSolver_6_3表示：位姿 _PoseDim 为6维，路标点 _LandmarkDim 是3维
  g2o::SparseOptimizer optimizer;
  g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

  linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

  g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType>(linearSolver));
  // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(std::unique_ptr<g2o::BlockSolver_6_3>(solver_ptr));
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<g2o::BlockSolver_6_3>(solver_ptr));
  optimizer.setAlgorithm(solver);
  // optimizer.setVerbose( true );       // 打开调试输出

  // 输入的帧中,有效的,参与优化过程的LINE-POINT对
  int nInitialCorrespondences=0;


  // // Step 2：添加顶点：待优化当前帧的Tcw
  VertexSE3Expmap* vSE3 = new VertexSE3Expmap();
  vSE3->setEstimate(Converter::toSE3Quat(mTge));
    // 设置id
  vSE3->setId(0);    
  // 要优化的变量，所以不能固定
  vSE3->setFixed(false);
  optimizer.addVertex(vSE3);

  //3D点的数量
  const int Nl = pointset_l.size();
  const int Nr = pointset_r.size();

  vector<MyEdgeSE3ProjectXYZOnlyPose*> vpEdges;
  vector<size_t> vnIndexEdge;
  vpEdges.reserve(Nl+Nr);
  vnIndexEdge.reserve(Nl+Nr);

  const float delta = sqrt(5.991); // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991

  // Step 3：添加一元边
  // 遍历当前地图中的所有地图点
  cv::Mat imagel = cv::imread("/media/zhutianyi/KESU/datasets/av2_vis/02678d04-cc9f-3148-9f95-1ba66347dff9/69/315969911357428266.jpg");
  //左侧相机
  for(int i=0; i<Nl; i++)
  {
    double w_sigma=1.0;
    Point3D &pMP = pointset_l[i];
    Point2D &npMP = pointset_l[i].npoint;

    // 对这个地图点的观测,是对应的线段的上距离地图点最近的2D点
    Eigen::Matrix<double,2,1> obs;
    obs(0,0) = npMP.x;
    obs(1,0) = npMP.y;
    //新建一元边
    MyEdgeSE3ProjectXYZOnlyPose* e = new MyEdgeSE3ProjectXYZOnlyPose();

    e->setId(i);
    // 设置边的顶点
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    e->setMeasurement(obs);
    // 这个点的可信程度和特征点所在的图层有关
    e->setInformation(Eigen::Matrix2d::Identity()*w_sigma);
    // 在这里使用了鲁棒核函数
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(delta);

    //添加内参，地图点Xw
    e->fx = fx_l;
    e->fy = fy_l;
    e->cx = cx_l;
    e->cy = cy_l;

    Matrix<double,4,4> extrincs = Converter::toMatrix4d(mego2caml);
    e->extrincs = extrincs;

    e->Xw[0] = pMP.x;
    e->Xw[1] = pMP.y;
    e->Xw[2] = pMP.z;

    // 
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    Vector3d pt3d(pMP.x, pMP.y, pMP.z);
    Vector4d pt3d_ego(vSE3_recov->estimate().map(pt3d)(0),vSE3_recov->estimate().map(pt3d)(1),vSE3_recov->estimate().map(pt3d)(2), 1.0);
    Eigen::Matrix<double,4,4> eigen_mego2caml;
    eigen_mego2caml = Converter::toMatrix4d(mego2caml);
    Vector4d pt3d_cam_ = eigen_mego2caml * pt3d_ego;
    Vector3d pt3d_cam(pt3d_cam_[0],pt3d_cam_[1],pt3d_cam_[2]);
    Vector2d pt2d = left_cam_project(pt3d_cam);
    //

    //如果3D点的投影在图像外，就不要了
    if((pt2d[0]>=0) && (pt2d[0]<=2048) && (pt2d[1]>=0) && (pt2d[1]<=1550))
    {
      nInitialCorrespondences++;
      optimizer.addEdge(e);
      vpEdges.push_back(e);
      vnIndexEdge.push_back(i);    
      // e->computeError();
      // const float chi2 = e->chi2();
      // cout << std::to_string(i)+": "  << chi2 << endl;  
      // cout << "pMP: " << pMP.x <<" " <<pMP.y << " " <<pMP.z << endl;
      // cout << pt2d[0]<< " " << pt2d[1] << endl;
      cv::Point point2d(pt2d[0], pt2d[1]); // 指定点的坐标 (x, y)
      cv::circle(imagel, point2d, 5, cv::Scalar(0, 0, 255), -1); // 5是圆的半径，Scalar是颜色，-1表示填充
      // cout << npMP.x << " " << npMP.y << endl;
      cv::Point mpoint2d(npMP.x, npMP.y); // 指定点的坐标 (x, y)
      cv::circle(imagel, mpoint2d, 5, cv::Scalar(0, 255, 0), -1); // 5是圆的半径，Scalar是颜色，-1表示填充 
    }
  }
  cv::imwrite("/media/zhutianyi/KESU/datasets/av2_vis/02678d04-cc9f-3148-9f95-1ba66347dff9/69/l_stereo.jpg", imagel);

  cv::Mat imager = cv::imread("/media/zhutianyi/KESU/datasets/av2_vis/02678d04-cc9f-3148-9f95-1ba66347dff9/69/315969911342441192.jpg");
  // 右侧相机
  for(int i=0; i<Nr; i++)
  {
    double w_sigma=1.0;
    Point3D pMP = pointset_r[i];
    Point2D npMP = pointset_r[i].npoint;

    // 对这个地图点的观测,是对应的线段的上距离地图点最近的2D点
    Eigen::Matrix<double,2,1> obs;
    obs(0,0) = npMP.x;
    obs(1,0) = npMP.y;
    //新建一元边
    MyEdgeSE3ProjectXYZOnlyPose* e = new MyEdgeSE3ProjectXYZOnlyPose();

    e->setId(i);
    // 设置边的顶点
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    e->setMeasurement(obs);
    // 这个点的可信程度和特征点所在的图层有关
    e->setInformation(Eigen::Matrix2d::Identity()*w_sigma);
    // 在这里使用了鲁棒核函数
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(delta);

    //添加内参，地图点Xw
    e->fx = fx_r;
    e->fy = fy_r;
    e->cx = cx_r;
    e->cy = cy_r;

    Matrix<double,4,4> extrincs = Converter::toMatrix4d(mego2camr);
    e->extrincs = extrincs;

    e->Xw[0] = pMP.x;
    e->Xw[1] = pMP.y;
    e->Xw[2] = pMP.z;

    // 
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    Vector3d pt3d(pMP.x, pMP.y, pMP.z);
    Vector4d pt3d_ego(vSE3_recov->estimate().map(pt3d)(0),vSE3_recov->estimate().map(pt3d)(1),vSE3_recov->estimate().map(pt3d)(2), 1.0);
    Eigen::Matrix<double,4,4> eigen_mego2camr;
    eigen_mego2camr = Converter::toMatrix4d(mego2camr);
    Vector4d pt3d_cam_ = eigen_mego2camr * pt3d_ego;
    Vector3d pt3d_cam(pt3d_cam_[0],pt3d_cam_[1],pt3d_cam_[2]);
    Vector2d pt2d = right_cam_project(pt3d_cam);
    //
    //如果3D点的投影在图像外，就不要了
    if((pt2d[0]>=0) && (pt2d[0]<=2048) && (pt2d[1]>=0) && (pt2d[1]<=1550))
    {
      nInitialCorrespondences++;
      optimizer.addEdge(e);
      vpEdges.push_back(e);
      vnIndexEdge.push_back(i+Nl);  
      // e->computeError();    
      // const float chi2 = e->chi2();
      // cout << std::to_string(i+Nl)+": "  << chi2 << endl;
      // cout << "pMP: " << pMP.x <<" " <<pMP.y << " " <<pMP.z << endl;
      // cout << pt2d[0]<< " " << pt2d[1] << endl;
      cv::Point point2d(pt2d[0], pt2d[1]); // 指定点的坐标 (x, y)
      cv::circle(imager, point2d, 5, cv::Scalar(0, 0, 255), -1); // 5是圆的半径，Scalar是颜色，-1表示填充
      // cout << npMP.x << " " << npMP.y << endl;
      cv::Point mpoint2d(npMP.x, npMP.y); // 指定点的坐标 (x, y)
      cv::circle(imager, mpoint2d, 5, cv::Scalar(0, 255, 0), -1); // 5是圆的半径，Scalar是颜色，-1表示填充 
    }
  }
  cv::imwrite("/media/zhutianyi/KESU/datasets/av2_vis/02678d04-cc9f-3148-9f95-1ba66347dff9/69/r_stereo.jpg", imager);

  cout << "边的数量:" << nInitialCorrespondences << endl;

  //点太少了，放弃
  if(nInitialCorrespondences<3){
    cout << "too less points, give up." << endl;
    return pose;
  }

  // Step 4：开始优化，总共优化n次，每次优化迭代10次,每次优化后，重新匹配对应点
  const int n = 50;
  float mychi2[n]; 
  int its[n];// n次迭代，每次迭代10次.
  for (int m = 0; m < n; m++) {
      mychi2[m] = 5.991;
      its[m] = 5;
  }
  // bad 的地图点个数
  int nBad=0;
  // 一共进行四次优化
  for(size_t it=0; it<n; it++)
  {
    optimizer.initializeOptimization(0);
    optimizer.optimize(its[it]);

    g2o::VertexSE3Expmap* vSE3_now = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_now = vSE3_now->estimate();
    mTge = Converter::toCvMat(SE3quat_now);

    //按照新的顶点pose重新投影，更新对应关系
    SearchPairLeft(pointset_l,lineset_l);
    SearchPairRight(pointset_r,lineset_r);
    cv::Mat imagel = cv::imread("/media/zhutianyi/KESU/datasets/av2_vis/02678d04-cc9f-3148-9f95-1ba66347dff9/69/315969911357428266.jpg");
    cv::Mat imager = cv::imread("/media/zhutianyi/KESU/datasets/av2_vis/02678d04-cc9f-3148-9f95-1ba66347dff9/69/315969911342441192.jpg");
    
    nBad=0;
    // 优化结束,开始遍历参与优化的每一条误差边
    for(size_t i=0, iend=vpEdges.size(); i<iend; i++)
    {
      MyEdgeSE3ProjectXYZOnlyPose* e = vpEdges[i];
      const size_t idx = vnIndexEdge[i];

      Point3D  pMP;
      Point2D  npMP;
      if(idx<Nl){
        pMP = pointset_l[idx];
        npMP = pointset_l[idx].npoint;        
      }
      else{
        pMP = pointset_r[idx-Nl];
        npMP = pointset_r[idx-Nl].npoint; 
      }

      Eigen::Matrix<double,2,1> obs;
      obs(0,0) = npMP.x;
      obs(1,0) = npMP.y;
      //更新观测量
      e->setMeasurement(obs);

      // //判断外殿
      // if(pMP.isOutlier){
      //   e->computeError(); 
      // }
      // // 就是error*\Omega*error,表征了这个点的误差大小(考虑置信度以后)
      // const float chi2 = e->chi2();
      
      // if(chi2>mychi2[it])
      // {                
      //     pMP.isOutlier = true;
      //     e->setLevel(1);                 // 设置为outlier , level 1 对应为外点,上面的过程中我们设置其为不优化
      //     nBad++;
      // }
      // else
      // {
      //     pMP.isOutlier=false;
      //     e->setLevel(0);                 // 设置为inlier, level 0 对应为内点,上面的过程中我们就是要优化这些关系
      // }

      //vis      
      if(idx<Nl){
        Vector3d pt3d(pMP.x, pMP.y, pMP.z);
        Vector4d pt3d_ego(vSE3_now->estimate().map(pt3d)(0),vSE3_now->estimate().map(pt3d)(1),vSE3_now->estimate().map(pt3d)(2), 1.0);
        Eigen::Matrix<double,4,4> eigen_mego2caml;
        eigen_mego2caml = Converter::toMatrix4d(mego2caml);
        Vector4d pt3d_cam_ = eigen_mego2caml * pt3d_ego;
        Vector3d pt3d_cam(pt3d_cam_[0],pt3d_cam_[1],pt3d_cam_[2]);
        Vector2d pt2d = left_cam_project(pt3d_cam);
        // cout << pt2d[0] << " " << pt2d[1] << endl;
        cv::Point point2d(pt2d[0], pt2d[1]); // 指定点的坐标 (x, y)
        cv::circle(imagel, point2d, 5, cv::Scalar(0, 0, 255), -1); // 5是圆的半径，Scalar是颜色，-1表示填充
        // cout << npMP.x << " " << npMP.y << endl;
        cv::Point mpoint2d(npMP.x, npMP.y); // 指定点的坐标 (x, y)
        // cv::circle(imagel, mpoint2d, 5, cv::Scalar(0, 255, 0), -1); // 5是圆的半径，Scalar是颜色，-1表示填充 
      }
      else{
        //vis
        Vector3d pt3d(pMP.x, pMP.y, pMP.z);
        Vector4d pt3d_ego(vSE3_now->estimate().map(pt3d)(0),vSE3_now->estimate().map(pt3d)(1),vSE3_now->estimate().map(pt3d)(2), 1.0);
        Eigen::Matrix<double,4,4> eigen_mego2camr;
        eigen_mego2camr = Converter::toMatrix4d(mego2camr);
        Vector4d pt3d_cam_ = eigen_mego2camr * pt3d_ego;
        Vector3d pt3d_cam(pt3d_cam_[0],pt3d_cam_[1],pt3d_cam_[2]);
        Vector2d pt2d = right_cam_project(pt3d_cam);
        // cout << pt2d[0] << " " << pt2d[1] << endl;
        cv::Point point2d(pt2d[0], pt2d[1]); // 指定点的坐标 (x, y)
        cv::circle(imager, point2d, 5, cv::Scalar(0, 0, 255), -1); // 5是圆的半径，Scalar是颜色，-1表示填充
        // cout << npMP.x << " " << npMP.y << endl;
        cv::Point mpoint2d(npMP.x, npMP.y); // 指定点的坐标 (x, y)
        // cv::circle(imager, mpoint2d, 5, cv::Scalar(0, 255, 0), -1); // 5是圆的半径，Scalar是颜色，-1表示填充 
      }
    }
    cv::imwrite("/media/zhutianyi/KESU/datasets/av2_vis/02678d04-cc9f-3148-9f95-1ba66347dff9/69/l_stereo_" + std::to_string(it) + ".jpg", imagel);
    cv::imwrite("/media/zhutianyi/KESU/datasets/av2_vis/02678d04-cc9f-3148-9f95-1ba66347dff9/69/r_stereo_" + std::to_string(it) + ".jpg", imager);

    // if(optimizer.edges().size()<10){
    //     cout << "too less edges." << endl;
    //     break;}
  }
    
  int nGood =  nInitialCorrespondences - nBad;
  cout << "nGood:" << nGood << endl;
  cout << "nBad:" << nBad << endl;

  // Step 5 得到优化后的当前帧的位姿
  g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
  g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
  pose = Converter::toCvMat(SE3quat_recov);
  cout << pose << endl;
  return pose;
}

Vector2d project2d(const Vector3d& v)  {
  Vector2d res;
  res(0) = v(0)/v(2);
  res(1) = v(1)/v(2);
  return res;
}

//从自车坐标系变换到像素坐标系
Vector2d MyEdgeSE3ProjectXYZOnlyPose::cam_project(const Vector3d & trans_xyz) const{
  Vector4d pte(trans_xyz[0],trans_xyz[1],trans_xyz[2],1.0);
  Eigen::Vector4d ptc = extrincs * pte;
  Eigen::Vector3d pt_cam(ptc[0],ptc[1],ptc[2]);
  Vector2d proj = project2d(pt_cam);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

// //从相机坐标系变换到像素坐标系
// Vector2d MyEdgeSE3ProjectXYZOnlyPose::cam_project(const Vector3d & trans_xyz) const{
//   Vector2d proj = project2d(trans_xyz);
//   Vector2d res;
//   res[0] = proj[0]*fx + cx;
//   res[1] = proj[1]*fy + cy;
//   return res;
// }


void MyEdgeSE3ProjectXYZOnlyPose:: linearizeOplus() {
  VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw);//自车坐标系

  Vector4d pte(xyz_trans[0],xyz_trans[1],xyz_trans[2],1.0);
  Eigen::Vector4d ptc = extrincs * pte;
  Eigen::Vector3d xyz_cam(ptc[0],ptc[1],ptc[2]);//相机坐标系

  double xe = xyz_trans[0];
  double ye = xyz_trans[1];
  double ze = xyz_trans[2];

  double x = xyz_cam[0];
  double y = xyz_cam[1];
  double z = xyz_cam[2];
  double invz = 1.0/xyz_cam[2];
  double invz_2 = invz*invz;

  double t00 = extrincs(0,0);
  double t01 = extrincs(0,1);
  double t02 = extrincs(0,2);
  double t10 = extrincs(1,0);
  double t11 = extrincs(1,1);
  double t12 = extrincs(1,2);
  double t20 = extrincs(2,0);
  double t21 = extrincs(2,1);
  double t22 = extrincs(2,2);

  _jacobianOplusXi(0,0) = (-fx*invz)*(-t01*ze+t02*ye) + fx*x*invz_2*(-t21*ze+t22*ye);
  _jacobianOplusXi(0,1) = (-fx*invz)*(t00*ze-t02*xe) + fx*x*(invz_2)*(t20*ze-t22*xe);
  _jacobianOplusXi(0,2) = (-fx*invz)*(-t00*ye+t01*xe) + fx*x*(invz_2)*(-t20*ye+t21*xe);
  _jacobianOplusXi(0,3) = (-fx*invz)*t00 + fx*x*(invz_2)*t20;
  _jacobianOplusXi(0,4) = (-fx*invz)*t01 + fx*x*(invz_2)*t21;
  _jacobianOplusXi(0,5) = (-fx*invz)*t02 + fx*x*(invz_2)*t22;

  _jacobianOplusXi(1,0) = (-fy*invz)*(-t11*ze+t12*ye) + (fy*y*invz_2)*(-t11*ze+t22*ye);
  _jacobianOplusXi(1,1) = (-fy*invz)*(t10*ze-t12*xe) + (fy*y*invz_2)*(t20*ze-t22*xe);
  _jacobianOplusXi(1,2) = (-fy*invz)*(-t10*ye+t11*xe) + (fy*y*invz_2)*(-t20*ye+t21*xe);
  _jacobianOplusXi(1,3) = (-fy*invz)*t10 + (fy*y*invz_2)*t20;
  _jacobianOplusXi(1,4) = (-fy*invz)*t11 + (fy*y*invz_2)*t21;
  _jacobianOplusXi(1,5) = (-fy*invz)*t12 + (fy*y*invz_2)*t22;

 

  //原来的雅可比
  // _jacobianOplusXi(0,0) =  x*y*invz_2 *fx;
  // _jacobianOplusXi(0,1) = -(1+(x*x*invz_2)) *fx;
  // _jacobianOplusXi(0,2) = y*invz *fx;
  // _jacobianOplusXi(0,3) = -invz *fx;
  // _jacobianOplusXi(0,4) = 0;
  // _jacobianOplusXi(0,5) = x*invz_2 *fx;

  // _jacobianOplusXi(1,0) = (1+y*y*invz_2) *fy;
  // _jacobianOplusXi(1,1) = -x*y*invz_2 *fy;
  // _jacobianOplusXi(1,2) = -x*invz *fy;
  // _jacobianOplusXi(1,3) = 0;
  // _jacobianOplusXi(1,4) = -invz *fy;
  // _jacobianOplusXi(1,5) = y*invz_2 *fy;
}