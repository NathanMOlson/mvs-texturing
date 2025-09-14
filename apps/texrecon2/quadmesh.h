#ifndef __QUADMESH_H__
#define __QUADMESH_H__

#include "math/vector.h"
#include <opencv2/core.hpp>
#include <filesystem>

class QuadMesh
{
public:
    QuadMesh(const std::filesystem::path& filepath);
    size_t NumFaces() const;
    size_t NumFaceRows() const;
    size_t NumFaceCols() const;
    math::Vec3f GetVertex(size_t i, size_t j) const;

private:
    cv::Mat _image;
    double _geo_transform[6];
};

#endif
