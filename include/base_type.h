#ifndef BASE_TYPE_H
#define BASE_TYPE_H
#include <vector>

using namespace std;
//实现功能：对于pointset中的每个Point2D实例（也就是2D点），计算该点到lineset中所有Line的距离，并找到与该点距离最短的Line，记录对应关系
class Point2D
{
public:
    Point2D(){}

    Point2D(const float &xin, const float &yin) : x(xin), y(yin){}

    Point2D(const Point2D& pt):x(pt.x),y(pt.y){}

    float x = 0.0;
    float y = 0.0;
};

class Point3D
{
public:
    Point3D(){}

    Point3D(const float &xin, const float &yin, const float &zin) : x(xin), y(yin), z(zin){}

    Point3D(const Point3D& pt):x(pt.x),y(pt.y),z(pt.z){}

    int index = -999; ///
    float x = 0.0;
    float y = 0.0;
    float z = 0.0;
    bool isOutlier = false;
    Point2D  npoint;
};

class Line
{
public:
    Line(const Point2D& pt1, const Point2D& pt2)
    {
        points.push_back(pt1);
        points.push_back(pt2);
    }

    vector<Point2D> points;
    int index = -999; ///
};

#endif