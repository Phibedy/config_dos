#define USE_OPENCV
#include "simple_visual_odometry.h"
#include <opencv/cv.h>
#include "vo_features.h"
#include <lms/imaging/graphics.h>

bool SimpleVisualOdometry::initialize() {
    image = readChannel<lms::imaging::Image>("IMAGE");
    debugImage = writeChannel<lms::imaging::Image>("DEBUG_IMAGE");
    configsChanged();
    return true;
}

bool SimpleVisualOdometry::deinitialize() {
    return true;
}

bool SimpleVisualOdometry::cycle() {
    bool drawDebug = config().get<bool>("drawDebug",false);
    //catch first round
    if(oldImage.width()==0 || oldImage.height() == 0){
        //first round
        oldImage = *image;
        return true;
    }
    //clear tmp objects
    newPoints.clear();
    status.clear();

    cv::Mat oldIm = oldImage.convertToOpenCVMat();

    int minFeatureCount = config().get<int>("minFeatureCount",1);
    if(oldPoints.size() <minFeatureCount){
        oldPoints.clear();
        featureDetection(oldIm, oldPoints); //detect points
        if(oldPoints.size() == 0){
            logger.error("No features detected!");
            return false;
        }
    }

    cv::Mat newIm = image->convertToOpenCVMat();
    if(drawDebug){
        debugImage->resize(image->width(),image->height(),lms::imaging::Format::BGRA);
        debugImage->fill(0);
    }

    logger.error("oldPoints")<<oldPoints.size();
    featureTracking(oldIm,newIm,oldPoints,newPoints, status); //track those features to the new image

    if(drawDebug){
        lms::imaging::BGRAImageGraphics graphics(*debugImage);
        graphics.setColor(lms::imaging::red);
        for(cv::Point2f p:newPoints){
            graphics.drawCross(p.x,p.y);
        }
    }

    //transform points to 2D-Coordinates
    std::vector<cv::Point2f> world_old,world_new;

    cv::perspectiveTransform(oldPoints,world_old,cam2world);
    cv::perspectiveTransform(newPoints,world_new,cam2world);
    //######################################################
    /*
    //from http://math.stackexchange.com/questions/77462/finding-transformation-matrix-between-two-2d-coordinate-frames-pixel-plane-to-w
    //create data
    cv::Mat leftSide,rightSide;
    rightSide.create(2*newPoints.size(),1, CV_64F);
    leftSide.create(2*newPoints.size(),4,CV_64F);
    for(std::size_t i = 0; i < 2*newPoints.size(); i+=2){
        leftSide.at<double>(i,0) = newPoints[i/2].x;
        leftSide.at<double>(i,1) = -newPoints[i/2].y;
        leftSide.at<double>(i,2) = 1;
        leftSide.at<double>(i,3) = 0;
        leftSide.at<double>(i+1,0) = newPoints[i/2+1].y;
        leftSide.at<double>(i+1,1) = newPoints[i/2+1].x;
        leftSide.at<double>(i+1,2) = 0;
        leftSide.at<double>(i+1,3) = 1;
        rightSide.at<double>(i,0) = newPoints[i/2].x;
        rightSide.at<double>(i+1,0) = newPoints[i/2].y;
    }
    //solve it
    cv::Mat res;
    cv::solve(leftSide,rightSide,res); //TODO we could use pseudo-inverse
    */

    //logger.error("result")<<res;

    //TODO We could try Kabasch_algoithm

    //set old values
    oldImage = *image;
    oldPoints = newPoints;

    //cv::namedWindow( "Camera", WINDOW_AUTOSIZE );

    return true;
}

void SimpleVisualOdometry::configsChanged(){
    cam2world.create(3,3,CV_64F);
    std::vector<float> points = config().getArray<float>("cam2world");
    if(points.size() != 9){
        logger.error("invalid cam2world");
        return;
    }
    int i = 0;
    for(int r = 0; r < 3; r++) {
        for(int c = 0; c < 3; c++) {
            cam2world.at<double>(r, c) = points[i];
            i++;
        }
    }
}
