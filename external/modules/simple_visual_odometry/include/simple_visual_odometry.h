#ifndef SIMPLE_VISUAL_ODOMETRY_H
#define SIMPLE_VISUAL_ODOMETRY_H

#include <lms/module.h>
#define USE_OPENCV
#include <lms/imaging/image.h>

/**
 * @brief LMS module simple_visual_odometry
 **/
class SimpleVisualOdometry : public lms::Module {
    lms::ReadDataChannel<lms::imaging::Image> image;
    lms::WriteDataChannel<lms::imaging::Image> debugImage;
    lms::imaging::Image oldImage;
    std::vector<cv::Point2f> oldPoints;

    cv::Mat world2cam,cam2world;

    //tmp objects
    std::vector<cv::Point2f> newPoints;
    std::vector<uchar> status;

public:
    bool initialize() override;
    bool deinitialize() override;
    bool cycle() override;
    void configsChanged();
};

#endif // SIMPLE_VISUAL_ODOMETRY_H
