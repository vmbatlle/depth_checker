#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "npy.hpp"
#include "opencv2/opencv.hpp"

constexpr float SCALE_FACTOR = 5.4;
constexpr float MIN_DEPTH = 1e-3;
constexpr float MAX_DEPTH = 800.0;
constexpr int IMG_WIDTH = 1241;
constexpr int IMG_HEIGHT = 376;

void cvMatFromNumpy(std::string path, cv::Mat& mat) {
    std::vector<unsigned long> shape;
    bool fortran_order;
    std::vector<float> data;

    shape.clear();
    data.clear();
    npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
    mat.create(shape[2], shape[3], CV_32F);
    float* pMat = mat.ptr<float>();
    for (float d : data) {
        *pMat = d;
        ++pMat;
    }
}

/**
 * Returns the q-th percentile of the matrix
 */
template <typename T>
T percentile(const cv::Mat& m, float q) {
    cv::Mat mat = m.clone();
    int nel = mat.rows * mat.cols;
    int nth = nel * q;
    std::nth_element(mat.begin<T>(), mat.begin<T>() + nth,
                     mat.end<T>());
    return *(mat.begin<T>() + nth);
}

/**
 * Shows heatmap of the disparity, normalized to [0.0, 1.0].
 */
void disp_show(const cv::Mat& disp) {
    double vmin;
    cv::minMaxLoc(disp, &vmin, nullptr);
    float vmax = percentile<float>(disp, 0.95);

    cv::Mat plot_disp;
    disp.convertTo(plot_disp, CV_8UC1, 255 / (vmax - vmin), vmin);
    cv::applyColorMap(plot_disp, plot_disp, cv::COLORMAP_MAGMA);
    cv::imshow("disparity heat map", plot_disp);
    cv::waitKey();
}

/**
 * Converts disparity (in pixels) to depth (in meters)
 * Uses a scale factor of 5.4 due to:
 *     A. KITTI baseline is 0.54m
 *     B. Monodepth2 NN train baseline is 0.1m
 */
void disp_to_depth(const cv::Mat& disp, cv::Mat& depth) {
    depth = 1.0f / disp;
    depth = depth * SCALE_FACTOR;
    cv::threshold(depth, depth, MIN_DEPTH, 0.0, cv::THRESH_TOZERO);
    cv::threshold(depth, depth, MAX_DEPTH, 0.0, cv::THRESH_TRUNC);
#ifdef DEBUG
    double min_depth, max_depth;
    cv::minMaxLoc(depth, &min_depth, &max_depth);
    std::cout << min_depth << " " << max_depth << std::endl;
#endif
}

/**
 * Shows heatmap of the depth, normalized to [0.0, 1.0].
 */
void depth_show(const cv::Mat& depth) {
    cv::Mat plot_depth;
    double vmin;
    cv::minMaxLoc(depth, &vmin, nullptr);
    float vmax = percentile<float>(depth, 0.95);
    depth.convertTo(plot_depth, CV_8UC1, 255 / (vmax - vmin), vmin);
    cv::Mat userColor(256, 1, CV_8UC1);
    for (int i = 0; i < 256; i++) userColor.at<uchar>(i) = 255 - i;
    cv::applyColorMap(userColor, userColor, cv::COLORMAP_MAGMA);
    cv::applyColorMap(plot_depth, plot_depth, userColor);
    cv::imshow("depth heat map", plot_depth);
    cv::waitKey();
}

int main(int, char** argv) {
    cv::Mat pred_disp, scaled_disp, depth;
    cvMatFromNumpy(argv[1], pred_disp);
    cv::resize(pred_disp, scaled_disp, cv::Size(IMG_WIDTH, IMG_HEIGHT));
#ifdef DEBUG
    disp_show(scaled_disp);
#endif
    disp_to_depth(scaled_disp, depth);
#ifdef DEBUG
    depth_show(depth);
#endif
    return 0;
}