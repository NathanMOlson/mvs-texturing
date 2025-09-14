#include "image_view.h"
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

void ImageView::initializeCameraPos(const math::Vec3f &trans, const math::Matrix3f &rot)
{
    pos[0] = -rot[0] * trans[0] - rot[3] * trans[1] - rot[6] * trans[2];
    pos[1] = -rot[1] * trans[0] - rot[4] * trans[1] - rot[7] * trans[2];
    pos[2] = -rot[2] * trans[0] - rot[5] * trans[1] - rot[8] * trans[2];
}

void ImageView::initializeViewDir(const math::Matrix3f &rot)
{
    viewdir[0] = rot[6];
    viewdir[1] = rot[7];
    viewdir[2] = rot[8];
}

void ImageView::initializeWorldToCam(const math::Vec3f &trans, const math::Matrix3f &rot)
{
    _world_to_cam[0] = rot[0];
    _world_to_cam[1] = rot[1];
    _world_to_cam[2] = rot[2];
    _world_to_cam[3] = trans[0];
    _world_to_cam[4] = rot[3];
    _world_to_cam[5] = rot[4];
    _world_to_cam[6] = rot[5];
    _world_to_cam[7] = trans[1];
    _world_to_cam[8] = rot[6];
    _world_to_cam[9] = rot[7];
    _world_to_cam[10] = rot[8];
    _world_to_cam[11] = trans[2];
    _world_to_cam[12] = 0.0f;
    _world_to_cam[13] = 0.0f;
    _world_to_cam[14] = 0.0f;
    _world_to_cam[15] = 1.0f;
}

math::Matrix3f get_rotation_matrix(const math::Vec3f &rotation)
{
    float len = rotation.norm();
    math::Matrix3f K;
    K[0] = 0;
    K[1] = -rotation[2] / len;
    K[2] = rotation[1] / len;
    K[3] = rotation[2] / len;
    K[4] = 0;
    K[5] = -rotation[0] / len;
    K[6] = -rotation[1] / len;
    K[7] = rotation[0] / len;
    K[8] = 0;
    math::Matrix3f I(0.F);
    I[0] = 1;
    I[4] = 1;
    I[8] = 1;
    return I + K * sin(len) + K * K * (1 - cos(len));
}

ImageView::ImageView(std::size_t id, const math::Vec3f &translation,
                     const math::Vec3f &rotation,
                     std::shared_ptr<Undistorter> undistorter,
                     const std::filesystem::path &image_file)
    : id(id), image_file(image_file), _undistorter(undistorter)
{
    math::Matrix3f rotation_matrix = get_rotation_matrix(rotation);
    initializeCameraPos(translation, rotation_matrix);
    initializeViewDir(rotation_matrix);
    initializeWorldToCam(translation, rotation_matrix);

    _weight_br = cv::Mat(_tile_width, _tile_width, CV_32F);
    for (size_t i = 0; i < _tile_width; i++)
    {
        float y = (i + 0.5) / _tile_width;
        for (size_t j = 0; j < _tile_width; j++)
        {
            float x = (j + 0.5) / _tile_width;
            _weight_br.at<float>(i, j) = x * y;
        }
    }
    _weight_br = _weight_br / cv::sum(_weight_br)[0];

    cv::rotate(_weight_br, _weight_tl, cv::ROTATE_180);
    cv::rotate(_weight_br, _weight_tr, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(_weight_br, _weight_bl, cv::ROTATE_90_CLOCKWISE);
}

cv::Point2f ImageView::get_pixel_coords(math::Vec3f const &vertex) const
{
    math::Vec3f ray_cam = _world_to_cam.mult(vertex, 1.0f);
    ray_cam /= ray_cam[2];
    return _undistorter->GetPixelCoords(cv::Point3f(ray_cam[0], ray_cam[1], ray_cam[2]));
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

cv::Mat ImageView::GetTile(const std::vector<cv::Point2f> &corners, int interp_type, int border_mode) const
{
    std::vector<cv::Point2f> tile_corners;
    tile_corners.push_back(cv::Point2f(-0.5, -0.5));
    tile_corners.push_back(cv::Point2f(_tile_width - 0.5, -0.5));
    tile_corners.push_back(cv::Point2f(_tile_width - 0.5, _tile_width - 0.5));
    tile_corners.push_back(cv::Point2f(-0.5, _tile_width - 0.5));

    cv::Mat warp = cv::getPerspectiveTransform(corners, tile_corners);
    cv::Mat tile(_tile_width, _tile_width, image.type());
    cv::warpPerspective(image, tile, warp, tile.size(), interp_type, border_mode);
    return tile;
}

bool ImageView::IsImageLoaded() const
{
    return !image.empty();
}

std::filesystem::path ImageView::ImagePath() const
{
    return image_file;
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
                              QuadInfo *face_info) const
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
    cv::Mat tile = GetTile(corners, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::Mat tile_f;
    tile.convertTo(tile_f, CV_32F);

    if (face_info->fully_visible)
    {
        face_info->num_valid_pixels = _tile_width * _tile_width;
        face_info->tl = _weight_tl.dot(tile_f);
        face_info->tr = _weight_tr.dot(tile_f);
        face_info->br = _weight_br.dot(tile_f);
        face_info->bl = _weight_bl.dot(tile_f);
        face_info->tl_w = 1;
        face_info->tr_w = 1;
        face_info->br_w = 1;
        face_info->bl_w = 1;
    }
    else
    {
        cv::Mat mask = GetTile(corners, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
        if (mask.type() != CV_8U)
        {
            mask.convertTo(mask, CV_8U);
        }
        face_info->num_valid_pixels = cv::countNonZero(tile);
        cv::Mat weight_tl, weight_tr, weight_br, weight_bl;
        _weight_tl.copyTo(weight_tl, mask);
        _weight_tr.copyTo(weight_tr, mask);
        _weight_br.copyTo(weight_br, mask);
        _weight_bl.copyTo(weight_bl, mask);

        face_info->tl_w = cv::sum(weight_tl)[0];
        face_info->tr_w = cv::sum(weight_tr)[0];
        face_info->br_w = cv::sum(weight_br)[0];
        face_info->bl_w = cv::sum(weight_bl)[0];

        face_info->tl = weight_tl.dot(tile_f);
        face_info->tr = weight_tr.dot(tile_f);
        face_info->br = weight_br.dot(tile_f);
        face_info->bl = weight_bl.dot(tile_f);
    }

    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Sobel(tile, grad_x, CV_32F, 1, 0);
    cv::Sobel(tile, grad_y, CV_32F, 0, 1);
    cv::multiply(grad_x, grad_x, grad_x);
    cv::multiply(grad_y, grad_y, grad_y);
    // cv::sqrt(grad_x + grad_y, grad_x);
    gmi = cv::mean(1 - 1 / ((grad_x + grad_y) / 32 + 1))[0];

    face_info->quality = gmi;
}

std::shared_ptr<Undistorter> create_undistorter_brown(const json &cam)
{
    double fx = cam["focal_x"];
    double fy = cam["focal_y"];
    double cx = cam["c_x"];
    double cy = cam["c_y"];
    size_t width = cam["width"];
    size_t height = cam["height"];
    std::vector<double> dist_coeffs;
    dist_coeffs.push_back(cam["k1"]);
    dist_coeffs.push_back(cam["k2"]);
    dist_coeffs.push_back(cam["p1"]);
    dist_coeffs.push_back(cam["p2"]);
    dist_coeffs.push_back(cam["k3"]);
    return std::make_shared<Undistorter>(fx, fy, cx, cy, width, height, dist_coeffs);
}

std::shared_ptr<Undistorter> create_undistorter_perspective(const json &cam)
{
    double f = cam["focal"];
    size_t width = cam["width"];
    size_t height = cam["height"];
    std::vector<double> dist_coeffs;
    dist_coeffs.push_back(cam["k1"]);
    dist_coeffs.push_back(cam["k2"]);
    dist_coeffs.push_back(0);
    dist_coeffs.push_back(0);
    return std::make_shared<Undistorter>(f, f, width / 2.0 - 0.5, height / 2.0 - 0.5, width, height, dist_coeffs);
}

std::shared_ptr<Undistorter> create_undistorter(const json &cam)
{
    if (cam["projection_type"] == "brown")
    {
        return create_undistorter_brown(cam);
    }
    else if (cam["projection_type"] == "perspective")
    {
        return create_undistorter_perspective(cam);
    }
    throw std::invalid_argument("invalid projection type");
}

std::vector<ImageView> generate_image_views(const std::filesystem::path &json_file)
{
    std::ifstream f(json_file);
    json data = json::parse(f);

    std::vector<ImageView> image_views;
    std::map<std::string, std::shared_ptr<Undistorter>> undistorters;

    for (auto &[key, value] : data[0]["cameras"].items())
    {
        undistorters[key] = create_undistorter(value);
    }

    int i = 0;
    for (auto &[key, value] : data[0]["shots"].items())
    {
        math::Vec3f translation(value["translation"][0], value["translation"][1], value["translation"][2]);
        math::Vec3f rotation(value["rotation"][0], value["rotation"][1], value["rotation"][2]);
        std::filesystem::path image_path = json_file.parent_path() / std::filesystem::path("images") / key;
        image_views.push_back(ImageView(i, translation, rotation, undistorters[value["camera"]], image_path));

        i++;
    }
    return image_views;
}