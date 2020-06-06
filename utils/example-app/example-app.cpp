/**
* This file is part of a Bachelor's Degree Final Project.
*
* Copyright (C) 2020 Víctor M. Batlle <736478 at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/vmbatlle/depth_checker>
*
* This file is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This file is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this file. If not, see <http://www.gnu.org/licenses/>.
*
* Special thanks to <https://github.com/ZHEQIUSHUI> for having a public
* repository <https://github.com/ZHEQIUSHUI/monodepth2-cpp> whit the
* basics needed to port Monodepth2's Python NN to C++.
*
* See Monodepthd2 repo at <https://github.com/nianticlabs/monodepth2>
*/

#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

constexpr float SCALE_FACTOR = 5.4;
constexpr float SIGM_MIN_DEPTH = 0.1;
constexpr float SIGM_MAX_DEPTH = 100.0;
constexpr float MIN_DEPTH = 1e-3;
constexpr float MAX_DEPTH = 800.0;
constexpr int IMG_WIDTH = 1241;
constexpr int IMG_HEIGHT = 376;

/**
 * Returns the q-th percentile of the matrix
 */
template <typename T>
T percentile(const cv::Mat& m, float q) {
    cv::Mat mat = m.clone();
    int nel = mat.rows * mat.cols;
    int nth = nel * q;
    std::nth_element(mat.begin<T>(), mat.begin<T>() + nth, mat.end<T>());
    return *(mat.begin<T>() + nth);
}

/**
 * Shows heatmap of the disparity, normalized to [0.0, 1.0].
 */
void disp_show(const cv::Mat& disp, cv::Mat& im) {
    double vmin;
    cv::minMaxLoc(disp, &vmin, nullptr);
    float vmax = percentile<float>(disp, 0.95);

    disp.convertTo(im, CV_8UC1, 255 / (vmax - vmin), vmin);
    cv::applyColorMap(im, im, cv::COLORMAP_MAGMA);
}

void disp_show(const cv::Mat& disp) {
    cv::Mat plot_disp;
    disp_show(disp, plot_disp);
    cv::imshow("disparity heat map", plot_disp);
    cv::waitKey();
}

/**
 * Convert network's sigmoid output into depth prediction
 * The formula for this conversion is given in the 'additional considerations'
 * section of the paper from Niantic's Monodepth2.
 */
void sigm_to_disp(const cv::Mat& sigm, float min_depth, float max_depth,
                  cv::Mat& disp) {
    float min_disp = 1 / max_depth;
    float max_disp = 1 / min_depth;
    disp = min_disp + (max_disp - min_disp) * sigm;
}

/**
 * Converts disparity (in pixels) to depth (in meters)
 * Uses a scale factor of 5.4 due to:
 *     A. KITTI baseline is 0.54m
 *     B. Monodepth2 NN train baseline is 0.1m
 */
void disp_to_depth(const cv::Mat& disp, float scale, float min_depth,
                   float max_depth, cv::Mat& depth) {
    depth = 1.0f / disp;
    depth = depth * scale;
    cv::threshold(depth, depth, min_depth, 0.0, cv::THRESH_TOZERO);
    cv::threshold(depth, depth, max_depth, 0.0, cv::THRESH_TRUNC);
}

int main(int argc, char** argv) {
    torch::jit::script::Module encoder = torch::jit::load("encoder.cpt");
    encoder.to(at::kCUDA);
    torch::jit::script::Module decoder = torch::jit::load("decoder.cpt");
    decoder.to(at::kCUDA);
    cv::Mat src = cv::imread("000000.png");
    cv::Mat input_mat;
    int w = 1024;
    int h = 320;
    cv::resize(src, input_mat, cv::Size(w, h));
    input_mat.convertTo(input_mat, CV_32FC3, 1. / 255.);
    torch::Tensor tensor_image = torch::from_blob(
        input_mat.data, {1, input_mat.rows, input_mat.cols, 3}, torch::kF32);
    tensor_image = tensor_image.permute({0, 3, 1, 2});
    tensor_image = tensor_image.to(at::kCUDA);

    std::vector<torch::IValue> batch;
    batch.push_back(tensor_image);
    auto result_encoder = encoder.forward(batch);
    batch.clear();
    batch.push_back(result_encoder);
    auto result_decoder = decoder.forward(batch);
    auto tensor_result = result_decoder.toTensor().to(at::kCPU);
    tensor_result = tensor_result.permute({0, 3, 2, 1});
    cv::Mat disp = cv::Mat(h, w, CV_32FC1, tensor_result.data_ptr());

    cv::resize(disp, disp, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    sigm_to_disp(disp, SIGM_MIN_DEPTH, SIGM_MAX_DEPTH, disp);
    cv::Mat show;
    disp_show(disp, show);
    src.push_back(show);
    cv::imshow("src", src);
    cv::waitKey();
    return 0;
}