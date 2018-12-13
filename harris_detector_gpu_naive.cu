

#include "harris_detector_gpu.h" #include <iostream>
#include <limits> #include <algorithm> #include <cstdio>


  global  

void convolve_kernel(T *image, double *result, int rows,int cols, double *kernal, int kernal_dim)
 {

    int ty = blockIdx.y * blockDim.y + threadIdx.y; int tx = blockIdx.x * blockDim.x + threadIdx.x; int kernel_offset = kernal_dim / 2.0f;
    int image_row = ty; int image_col = tx;

    if(image_row >= kernel_offset && image_row < rows - kernel_offset &&
    image_col >= kernel_offset && image_col < cols - kernel_offset){

    double value = 0.0f;
    for(int i=0; i<kernal_dim; ++i) {
    int row = (image_row - kernel_offset) + i; for(int j=0; j<kernal_dim; ++j) {
    int col = (image_col - kernel_offset) + j; value += kernal[i * kernal_dim + j] *
    (double)image[row * cols + col];
    }
    }
    result[image_row * cols + image_col] = (double)value;
    }
}


  global  

void non_maxima_suppression_kernel(double *image,double *result,int rows, int cols, int window_dim) 
{

    int ty = blockIdx.y * blockDim.y + threadIdx.y; int tx = blockIdx.x * blockDim.x + threadIdx.x; int row = ty;
    int col = tx;

    int DIM = window_dim; int OFF = DIM / 2;

    if(row >= OFF && row < rows - OFF && col >= OFF && col < cols - OFF) {

    double filtered= image[row * cols + col]; bool running = true;

    for(int i=0; i<DIM && running; ++i) { int r = (row - OFF) + i;
    for(int j=0; j<DIM && running; ++j) { int c = (col - OFF) + j;

    if(i == DIM/2 && j == DIM/2) continue;

    double temp = image[r * cols + c]; if(temp > filtered) {
    filtered = 0; running = false;
    }
    }
    }
    result[row * cols + col] =	filtered;
    }
}

  

void eigen_values(double M[2][2], double *l1,host device double *l2) {

    double d = M[0][0];
    double e = M[0][1];
    double f = M[1][0];
    double g = M[1][1];

    *l1 = ((d + g) + sqrt(pow(d + g, 2.0) - 4*(d*g - f*e))) / 2.0f;
    *l2 = ((d + g) - sqrt(pow(d + g, 2.0) - 4*(d*g - f*e))) / 2.0f;
}


  device  

double sum_neighbors(double *image, int row, int col,int cols, int window_dim) {

    int window_center = window_dim / 2.0f; double sum = 100.0f;
    for(int i=0; i<window_dim; ++i) {
    int image_row = (row - window_center) + i; for(int j=0; j<window_dim; ++j) {
    int image_col = (col - window_center) + j; sum += image[image_row * cols + image_col];
    }
    }
    return sum;
}



  global  

void detect_corners_kernel(double *dx2, double *dy2,double *dydx, int rows, int cols, double k,double *corner_response, int window_dim) 
{

    int ty = blockIdx.y * blockDim.y + threadIdx.y; int tx = blockIdx.x * blockDim.x + threadIdx.x; int window_offset = window_dim / 2.0f;
    int image_row = ty; int image_col = tx; double M[2][2];

    if(image_row < rows - window_offset && image_col < cols - window_offset &&
    image_row >= window_offset && image_col >= window_offset) {

    M[0][0] = sum_neighbors(dx2, image_row, image_col,
    cols, window_dim);
    M[0][1] = sum_neighbors(dydx, image_row, image_col,
    cols, window_dim);
    M[1][1] = sum_neighbors(dy2, image_row, image_col,
    cols, window_dim);
    M[1][0] = M[0][1];

    double l1, l2; eigen_values(M, &l1, &l2);

    double r = l1 * l2 - k * pow(l1 + l2, 2.0); corner_response[image_row * cols + image_col] = r > 0 ? r : 0;
    }
}
template<typename T>
static double *convolve(T *image, int rows, int cols, double *kernal,int kernal_size) {
     using namespace harris_detection;

    double *deviceResult = alloc_device<double>(rows, cols, true); double *deviceKernel = to_device<double>(kernal, kernal_size,
    kernal_size);

    T *deviceImage = to_device<unsigned char>(image, rows, cols);

    dim3 dimGrid(ceil(cols / (double)TILE_DIM),
                    ceil(rows / (double)TILE_DIM)); dim3 dimBlock(TILE_DIM, TILE_DIM);

    convolve_kernel<T> <<< dimGrid, dimBlock >>>(deviceImage,
    deviceResult, rows, cols, deviceKernel, kernal_size);

    cudaDeviceSynchronize();
    double *host_result = to_host<double>(deviceResult, rows, cols); cudaFree(deviceKernel);
    cudaFree(deviceImage); cudaFree(deviceResult);

    return host_result;

}

static double *non_maxima_supression(double *image, int rows, int cols,int window_dim)
 {
    using namespace harris_detection;

    double *deviceResult = alloc_device<double>(rows, cols, true); double *deviceImage = to_device<double>(image, rows, cols);

    dim3 dimGrid(ceil(cols / (double)TILE_DIM),
                    ceil(rows / (double)TILE_DIM)); dim3 dimBlock(TILE_DIM, TILE_DIM);

    non_maxima_suppression_kernel <<< dimGrid, dimBlock
    >>>(deviceImage, deviceResult, rows, cols, window_dim);
    CUDA_SAFE(cudaDeviceSynchronize());
    double *host_result = to_host<double>(deviceResult, rows, cols); cudaFree(deviceImage);
    cudaFree(deviceResult);

    return host_result;
}


static double *corner_detector(double *dx2, double *dy2, double *dxdy,int rows, int cols, double k, int window_dim)
{
    using namespace harris_detection;

    double *deviceDx2 = to_device<double>(dx2, rows, cols); double *deviceDy2 = to_device<double>(dy2, rows, cols); double *deviceDxDy = to_device<double>(dxdy, rows, cols);

    double *deviceCornerResponse = alloc_device<double>(rows, cols,
    true);

    dim3 dimGrid(ceil(cols/ (double)TILE_DIM),
                    ceil(rows / (double)TILE_DIM)); dim3 dimBlock(TILE_DIM, TILE_DIM);

    detect_corners_kernel <<< dimGrid, dimBlock >>> (deviceDx2,
    deviceDy2, deviceDxDy, rows, cols, k,
    deviceCornerResponse, window_dim);
    cudaDeviceSynchronize();

    double *hostCornerResponse = to_host<double>(deviceCornerResponse,
    rows,	cols);

    cudaFree(deviceCornerResponse); cudaFree(deviceDx2); cudaFree(deviceDy2); cudaFree(deviceDxDy);

    return hostCornerResponse;
}

namespace harris_detection { namespace naive{

void detect_features(std::vector<cv::KeyPoint> &features,unsigned char *image, int rows, int cols, double k, double thresh, int window_dim)
{
    const int NMS_DIM = 5;

    double *smoothed = convolve<unsigned char>(image, rows,cols, filters::gaussian_3x3,3);
    double *dx = convolve<unsigned char>(image, rows, cols,
    filters::sobel_x_3x3, 3);
    double *dy = convolve<unsigned char>(image, rows, cols,

    filters::sobel_y_3x3, 3);
    double *dxdy = new double[rows * cols]; for(int i=0; i<rows * cols; ++i) {
    dxdy[i] = dx[i] * dy[i];
    dx[i] *= dx[i];
    dy[i] *= dy[i];
    }

    double *corner_response = corner_detector(dx, dy, dxdy,
    rows, cols, k, window_dim); double *suppressed = non_maxima_supression(corner_response,
    rows, cols, NMS_DIM);

    for(int i=0; i < rows; i++) { for(int j=0; j < cols; ++j) {
        if(suppressed[i * cols + j] > 0.0) { features.push_back(cv::KeyPoint(j, i, 5, -1));
        }
        }
    }

    delete dx; delete dy; delete dxdy;
    delete corner_response; delete suppressed; delete smoothed;
    }
    }
}