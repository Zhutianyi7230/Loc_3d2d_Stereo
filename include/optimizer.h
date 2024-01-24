#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <opencv2/core/core.hpp>
#include "base_type.h"
using namespace std;
using namespace Eigen;
using namespace g2o;


class Optimizer
{
public:
    Optimizer(){};
    Optimizer(cv::Mat &global_pose, cv::Mat &instrincsl, cv::Mat &instrincsr, cv::Mat &extrincsl, cv::Mat &extrincsr){
      mTge = global_pose;

      fx_l = instrincsl.at<double>(0,0);
      fy_l = instrincsl.at<double>(1,1);
      cx_l = instrincsl.at<double>(0,2);
      cy_l = instrincsl.at<double>(1,2);

      fx_r = instrincsr.at<double>(0,0);
      fy_r = instrincsr.at<double>(1,1);
      cx_r = instrincsr.at<double>(0,2);
      cy_r = instrincsr.at<double>(1,2);

      mego2caml = extrincsl;
      mego2camr = extrincsr;
    };

    Vector2d project2d(const Vector3d& v);
    Vector2d left_cam_project(const Vector3d & trans_xyz);
    Vector2d right_cam_project(const Vector3d & trans_xyz);

    void SearchPairLeft(vector<Point3D> &pointset, vector<Line> &lineset);
    void SearchPairRight(vector<Point3D> &pointset, vector<Line> &lineset);

    cv::Mat PoseOptimization(vector<Point3D> &pointset_l , vector<Point3D> &pointset_r, vector<Line> &lineset_l, vector<Line> &lineset_r);

    cv::Mat mTge = cv::Mat_<double>(4,4);//初始位姿（global2ego）
    double fx_l, fy_l, cx_l, cy_l;
    double fx_r, fy_r, cx_r, cy_r;
    cv::Mat mego2caml ;
    cv::Mat mego2camr ;
    cv::Mat pose = cv::Mat_<double>(4,4);
};




class  MyEdgeSE3ProjectXYZOnlyPose: public  g2o::BaseUnaryEdge<2, Vector2d, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MyEdgeSE3ProjectXYZOnlyPose(){}

  bool read(std::istream& is){
    for (int i=0; i<2; i++){
      is >> _measurement[i];
    }
    for (int i=0; i<2; i++)
      for (int j=i; j<2; j++) {
        is >> information()(i,j);
        if (i!=j)
          information()(j,i)=information()(i,j);
      }
    return true;
  }

  bool write(std::ostream& os) const{
    for (int i=0; i<2; i++){
      os << measurement()[i] << " ";
    }

    for (int i=0; i<2; i++)
      for (int j=i; j<2; j++){
        os << " " <<  information()(i,j);
      }
    return os.good();
  }

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    Vector2d obs(_measurement);
    _error = obs-cam_project(v1->estimate().map(Xw));

    // cout << Xw[0] << " " << Xw[1] << " " << Xw[2] << endl;
    // cout << cam_projec/t(v1->estimate().map(Xw))(0) << " " << cam_project(v1->estimate().map(Xw))(1) << endl;
    // cout << obs[0] <<" " << obs[1] << endl;
    // cout << _error[0] << " " << _error[1] << endl;
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    Vector4d pte((v1->estimate().map(Xw))(0),(v1->estimate().map(Xw))(1),(v1->estimate().map(Xw))(2),1.0);
    Eigen::Vector4d ptc = extrincs * pte;
    return ptc(2)>0.0;
    // return (v1->estimate().map(Xw))(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d & trans_xyz) const;

  Eigen::Vector3d Xw;
  Eigen::Matrix<double,4,4> extrincs;
  double fx, fy, cx, cy;
};

#endif