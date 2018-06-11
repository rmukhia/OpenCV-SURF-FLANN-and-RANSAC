#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"

#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <numeric>

#define RAND_SEED  9
using namespace cv;
using namespace cv::xfeatures2d;

void readme()
{ std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl; }


/** @function main */
int main(int argc, char** argv)
{
    if( argc != 3 )
    { readme(); return -1; }

    Mat img_object = imread(argv[1]);
    Mat img_scene = imread(argv[2]);

    if (!img_object.data || !img_scene.data)
    {
        std::cout << " --(!) Error reading images " << std::endl; return -1;
    }

    //-- Step 1: Detect the key points using SURF Detector
    int minHessian = 400;

    // Use Ptr as in https://github.com/kyamagu/mexopencv/issues/154
    Ptr<SurfFeatureDetector> detector = SurfFeatureDetector::create(minHessian);

    std::vector<KeyPoint> keyPoints_1, keyPoints_2;

    detector->detect(img_object, keyPoints_1);
    detector->detect(img_scene, keyPoints_2);

    //-- Step 2: Calculate descriptors (feature vectors)
    // Based on http://docs.opencv.org/3.2.0/d9/d97/tutorial_table_of_content_features2d.html
    Ptr<SurfDescriptorExtractor> extractor = SurfDescriptorExtractor::create();

    Mat descriptors_1, descriptors_2;

    extractor->compute(img_object, keyPoints_1, descriptors_1);
    extractor->compute(img_scene, keyPoints_2, descriptors_2);

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match(descriptors_1, descriptors_2, matches);


    Mat img_matches;
    drawMatches(img_object, keyPoints_1, img_scene, keyPoints_2,
                matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //-- Show detected matches
    namedWindow("imageSrc", WINDOW_NORMAL);
    imshow("imageSrc", img_matches);
    // scale image to fit on screen
    resizeWindow("imageSrc", 800, 600);

    // the number of feature points are too large, lets prune randomly and show
    std::vector<int> matches_index(matches.size());
    std::iota(matches_index.begin(), matches_index.end(), 0);


    std::shuffle(matches_index.begin(), matches_index.end(), std::mt19937(RAND_SEED));

    matches_index.resize(20);

    std::vector< DMatch > pruned_matches;

    for(auto i:matches_index) {
        pruned_matches.push_back(matches[i]);
    }

    drawMatches(img_object, keyPoints_1, img_scene, keyPoints_2,
                pruned_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //-- Show detected matches
    namedWindow("imageSrcPruned", WINDOW_NORMAL);
    imshow("imageSrcPruned", img_matches);
    // scale image to fit on screen
    resizeWindow("imageSrcPruned", 800, 600);

    //-- Localize the object
    std::vector<Point2f> center;
    std::vector<Point2f> transform;

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between key points
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;

    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (matches[i].distance < 3 * min_dist)
        {
            good_matches.push_back(matches[i]);
        }
    }

    for (auto &good_match : good_matches) {
        //-- Get the key points from the good matches
        center.push_back(keyPoints_1[good_match.queryIdx].pt);
        transform.push_back(keyPoints_2[good_match.trainIdx].pt);
    }

    Mat mask;
    Mat H = findHomography(center, transform, RANSAC, 3, mask);


    std::cout << "Homography:\n" << H << std::endl;

    std::vector<Point2f> keyPoints_p_1, keyPoints_p_2_transform;
    std::vector< DMatch > homography_matches;
    std::cout << "Removed Outliers:" << std::endl;
    int j = 1;
    std::cout << std::fixed;
    std::cout << std::setprecision(0);
    for(int i =0;i < mask.rows ; i++) {
        if (mask.at<uchar>(i) == 0) {
            std::cout << "(" << center[i] << " - " << transform[i] << ")," << "\t" ;
            if ( j % 4 == 0) std::cout << std::endl;
            j++;
        } else {
            keyPoints_p_1.push_back(center[i]);
            keyPoints_p_2_transform.push_back(transform[i]);
        }
    }

    for(int i=0; i< keyPoints_p_1.size(); i++)
        homography_matches.emplace_back(i, i, 0);


    Mat img_transform_homography;
    warpPerspective(img_object, img_transform_homography, H, img_scene.size());
    Mat img_matches_homography;

    std::vector<KeyPoint> keyPoints_1_transform, keyPoints_2_transform;

    //perspectiveTransform(keyPoints_p_1, keyPoints_p_2_transform, H);
    KeyPoint::convert(keyPoints_p_1, keyPoints_1_transform);
    KeyPoint::convert(keyPoints_p_2_transform, keyPoints_2_transform);


    drawMatches(img_object, keyPoints_1_transform, img_transform_homography, keyPoints_2_transform,
                homography_matches, img_matches_homography, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    namedWindow("imageDst", WINDOW_NORMAL);
    imshow("imageDst", img_matches_homography);
    resizeWindow("imageDst", 800, 600);




    // the number of feature points are too large, lets prune randomly and show
    std::vector<int> matches_index2(homography_matches.size());
    std::iota(matches_index2.begin(), matches_index2.end(), 0);


    std::shuffle(matches_index2.begin(), matches_index2.end(), std::mt19937(RAND_SEED));

    matches_index2.resize(20);

    pruned_matches.clear();
    for(auto i:matches_index2) {
        pruned_matches.emplace_back(i,i,0);
    }

    drawMatches(img_object, keyPoints_1_transform, img_transform_homography, keyPoints_2_transform,
                pruned_matches, img_matches_homography, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //-- Show detected matches
    namedWindow("imageDstPruned", WINDOW_NORMAL);
    imshow("imageDstPruned", img_matches_homography);
    // scale image to fit on screen
    resizeWindow("imageDstPruned", 800, 600);

    waitKey(0);
    return 0;
}
