#include "image_view.h"
#include "tex/progress_counter.h"
#include <opencv2/imgcodecs.hpp>
#include <mve/bundle_io.h>
#include <mve/scene.h>
#include <mve/image.h>
#include <mve/image_io.h>

ImageView::ImageView(std::size_t id,
                     mve::CameraInfo const &camera,
                     const std::filesystem::path &image_file)
    : id(id), image_file(image_file)
{

    mve::image::ImageHeaders header;
    try
    {
        header = mve::image::load_file_headers(image_file);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Could not load image header of " << image_file << std::endl;
        std::cerr << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    camera.fill_calibration(*projection, header.width, header.height);
    camera.fill_camera_pos(*pos);
    camera.fill_viewing_direction(*viewdir);
    camera.fill_world_to_cam(*world_to_cam);
}

cv::Point2f ImageView::get_pixel_coords(math::Vec3f const &vertex) const
{
    math::Vec3f pixel = projection * world_to_cam.mult(vertex, 1.0f);
    pixel /= pixel[2];
    return cv::Point2f(pixel[0] - 0.5f, pixel[1] - 0.5f);
}

std::vector<cv::Point2f> ImageView::get_pixel_coords(const std::vector<math::Vec3f> &vertices) const
{
    std::vector<cv::Point2f> pixels;
    for (const auto &vertex : vertices)
    {
        pixels.push_back(get_pixel_coords(vertex));
    }
    return pixels;
}

cv::Mat ImageView::GetTile(const std::vector<cv::Point2f> &corners) const
{
    std::vector<cv::Point2f> tile_corners;
    tile_corners.push_back(cv::Point2f(-0.5, -0.5));
    tile_corners.push_back(cv::Point2f(_tile_width - 0.5, -0.5));
    tile_corners.push_back(cv::Point2f(_tile_width - 0.5, _tile_width - 0.5));
    tile_corners.push_back(cv::Point2f(-0.5, _tile_width - 0.5));

    cv::Mat warp = cv::getPerspectiveTransform(corners, tile_corners);
    cv::Mat tile(_tile_width, _tile_width, image.type());
    cv::warpPerspective(image, tile, warp, tile.size());
    return tile;
}

inline float quad_area(const std::vector<cv::Point2f> &corners)
{
    if (corners.size() != 4)
    {
        return 0.0;
    }
    return 0.5 * (corners[0].x * corners[1].y - corners[0].y * corners[1].x + corners[1].x * corners[2].y - corners[1].y * corners[2].x + corners[2].x * corners[3].y - corners[2].y * corners[3].x + corners[3].x * corners[0].y - corners[3].y * corners[0].x);
}

bool ImageView::valid_pixel(cv::Point2f pixel) const
{
    return pixel.x >= -0.5 && pixel.x <= image.cols - 0.5 && pixel.y >= -0.5 && pixel.y <= image.rows - 0.5;
}

std::size_t ImageView::get_id(void) const
{
    return id;
}

math::Vec3f ImageView::get_pos(void) const
{
    return pos;
}

math::Vec3f ImageView::get_viewing_direction(void) const
{
    return viewdir;
}

bool ImageView::inside(const std::vector<cv::Point2f> &corners) const
{
    for (const auto &corner : corners)
    {
        if (!valid_pixel(corner))
        {
            return false;
        }
    }
    return true;
}

bool ImageView::intersects(const std::vector<cv::Point2f> &corners) const
{
    for (const auto &corner : corners)
    {
        if (valid_pixel(corner))
        {
            return true;
        }
    }
    return false;
}

void ImageView::load_image(void)
{
    image = cv::imread(image_file, cv::IMREAD_ANYDEPTH | cv::IMREAD_UNCHANGED);
}

void ImageView::release_image(void)
{
    image.release();
}

void ImageView::get_face_info(const std::vector<cv::Point2f> &corners,
                              QuadInfo *face_info, tex::Settings const &settings) const
{
    assert(!image.empty());
    face_info->fully_visible = inside(corners);

    float area = quad_area(corners);

    if (area < std::numeric_limits<float>::epsilon())
    {
        face_info->quality = 0.0f;
        return;
    }

    float gmi = 0;
    cv::Mat tile = GetTile(corners);
    if (face_info->fully_visible)
    {
        face_info->num_valid_pixels = _tile_width * _tile_width;
    }
    else
    {
        face_info->num_valid_pixels = cv::countNonZero(tile);
    }
    if (settings.data_term == tex::DATA_TERM_GMI)
    {
        cv::Mat grad_x;
        cv::Mat grad_y;
        cv::Sobel(tile, grad_x, CV_32F, 1, 0);
        cv::Sobel(tile, grad_y, CV_32F, 0, 1);
        cv::multiply(grad_x, grad_x, grad_x);
        cv::multiply(grad_y, grad_y, grad_y);
        cv::sqrt(grad_x + grad_y, grad_x);
        gmi = cv::mean(grad_x)[0];
    }

    switch (settings.data_term)
    {
    case tex::DATA_TERM_AREA:
        face_info->quality = area;
        break;
    case tex::DATA_TERM_GMI:
        face_info->quality = gmi;
        break;
    }
}

std::vector<ImageView> generate_image_views(const std::filesystem::path &nvm_file,
                                            const std::filesystem::path &tmp_dir)
{
    std::vector<ImageView> image_views;
    std::vector<mve::NVMCameraInfo> nvm_cams;
    mve::Bundle::Ptr bundle = mve::load_nvm_bundle(nvm_file, &nvm_cams);
    mve::Bundle::Cameras &cameras = bundle->get_cameras();

    ProgressCounter view_counter("\tLoading", cameras.size());
#pragma omp parallel for
#if !defined(_MSC_VER)
    for (std::size_t i = 0; i < cameras.size(); ++i)
    {
#else
    for (std::int64_t i = 0; i < cameras.size(); ++i)
    {
#endif
        view_counter.progress<SIMPLE>();
        mve::CameraInfo &mve_cam = cameras[i];
        mve::NVMCameraInfo const &nvm_cam = nvm_cams[i];

        cv::Mat image = cv::imread(nvm_cam.filename);
        int const maxdim = std::max(image.cols, image.rows);
        mve_cam.flen = mve_cam.flen / static_cast<float>(maxdim);

#pragma omp critical
        image_views.push_back(ImageView(i, mve_cam, nvm_cam.filename));

        view_counter.inc();
    }
    return image_views;
}