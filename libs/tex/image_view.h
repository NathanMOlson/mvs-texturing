#ifndef __IMAGE_VIEW_H__
#define __IMAGE_VIEW_H__

#include "math/vector.h"
#include "math/matrix.h"
#include "quadmesh.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "tex/settings.h"
#include <mve/camera.h>

struct QuadInfo
{
    std::uint16_t view_id;
    float quality;
    bool fully_visible;

    bool operator<(QuadInfo const &other) const
    {
        return view_id < other.view_id;
    }
};

/**
 * Class representing a view with specialized functions for texturing.
 */
class ImageView
{
private:
    std::size_t id;

    math::Vec3f pos;
    math::Vec3f viewdir;
    math::Matrix3f projection;
    math::Matrix4f world_to_cam;
    std::string image_file;
    cv::Mat image;

public:
    /** Returns the id of the TexureView which is consistent for every run. */
    std::size_t get_id(void) const;

    /** Returns the 2D pixel coordinates of the given vertex projected into the view. */
    cv::Point2f get_pixel_coords(math::Vec3f const &vertex) const;
    std::vector<cv::Point2f> get_pixel_coords(const std::vector<math::Vec3f> &vertex) const;
    /** Returns the RGB pixel values [0, 1] for the given vertex projected into the view, calculated by linear interpolation. */
    //        math::Vec3f get_pixel_values(math::Vec3f const & vertex) const;

    /** Returns whether the pixel location is valid in this view.
     * The pixel location is valid if its inside the visible area and,
     * if a validity mask has been generated, all surrounding (integer coordinate) pixels are valid in the validity mask.
     */
    bool valid_pixel(cv::Point2f pixel) const;

    bool inside(const std::vector<cv::Point2f> &corners) const;
    bool intersects(const std::vector<cv::Point2f> &corners) const;

    /** Constructs a ImageView from the give mve::CameraInfo containing the given image. */
    ImageView(std::size_t id, mve::CameraInfo const &camera, const std::filesystem::path &image_file);

    cv::Mat GetTile(const std::vector<cv::Point2f> &corners) const;

    /** Returns the position. */
    math::Vec3f get_pos(void) const;
    /** Returns the viewing direction. */
    math::Vec3f get_viewing_direction(void) const;

    /** Loads the corresponding image. */
    void load_image(void);

    /** Releases the corresponding image. */
    void release_image(void);

    void get_face_info(const std::vector<cv::Point2f> &corners,
                       QuadInfo *face_info, tex::Settings const &settings) const;
};

std::vector<ImageView> generate_image_views(const std::filesystem::path &nvm_file,
                                            const std::filesystem::path &tmp_dir);

#endif
