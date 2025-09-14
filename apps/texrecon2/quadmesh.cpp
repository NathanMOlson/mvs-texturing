#include "quadmesh.h"
#include <opencv2/imgcodecs.hpp>
#include <gdal/gdal_priv.h>

QuadMesh::QuadMesh(const std::filesystem::path &filepath)
{
    _image = cv::imread(filepath, cv::IMREAD_ANYDEPTH);
    if (_image.empty())
    {
        throw std::invalid_argument("Invalid file");
    }

    GDALDataset *dataset = (GDALDataset *)GDALOpen(filepath.c_str(), GA_ReadOnly);
    if (dataset == nullptr)
    {
        throw std::invalid_argument("Invalid file");
    }

    if (dataset->GetGeoTransform(_geo_transform) != CE_None)
    {
        GDALClose(dataset);
        throw std::invalid_argument("Invalid file");
    }
    GDALClose(dataset);
}

size_t QuadMesh::NumFaces() const
{
    return NumFaceRows() * NumFaceCols();
}

size_t QuadMesh::NumFaceRows() const
{
    return _image.rows - 1;
}

size_t QuadMesh::NumFaceCols() const
{
    return _image.cols - 1;
}

#include <iostream>

math::Vec3f QuadMesh::GetVertex(size_t i, size_t j) const
{
    return math::Vec3f(_geo_transform[0] + j * _geo_transform[1] + i * _geo_transform[2],
                       _geo_transform[3] + j * _geo_transform[4] + i * _geo_transform[5],
                       _image.at<float>(i, j));
}
