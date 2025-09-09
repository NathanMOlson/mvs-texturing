#ifndef __UNDISTORTER_H__
#define __UNDISTORTER_H__

#include <vector>
#include <opencv2/core.hpp>

class Undistorter
{
public:
    Undistorter(double fx, double fy, double cx, double cy, size_t width, size_t height, const std::vector<double> &dist_coeffs);

    std::vector<cv::Point2f> GetPixelCoords(const std::vector<cv::Point3f> &rays_cam) const;
    cv::Point2f GetPixelCoords(const cv::Point3f &ray_cam) const;
    cv::Mat Undistort(const cv::Mat &img) const;

private:
    cv::Mat _cam_old;
    cv::Mat _cam_new;
    std::vector<double> _dist_coeffs;

    cv::Mat _map1;
    cv::Mat _map2;
};

#endif
