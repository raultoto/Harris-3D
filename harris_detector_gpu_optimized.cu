


#include "harris_detector_gpu.h" #include <iostream>
#include <limits> #include <algorithm> #include <thrust/scan.h>
#include <thrust/functional.h> #include <thrust/host_vector.h> #include <thrust/device_vector.h> #include <thrust/device_ptr.h> #include <thrust/sort.h>

// GPGPU device memory image pool size #define DEVICE_RESULT_COUNT 8

// Global GPGPU device to allocated once for optimization double *deviceKernel = NULL;
unsigned char *deviceImage = NULL;

// Array of Global GPGPU memory images to be reused double *deviceResult[DEVICE_RESULT_COUNT] = {NULL}; double *deviceResultTemp = NULL;

// Pointer to array of CUDA streams cudaStream_t *deviceStreams = NULL; int deviceStreamCount = 0;

// Scan keys used for integral image exclusive scan int *scanKeys = NULL;

// Scan keys for transpose exclusive scan int *scanKeysT = NULL;

// Array of scan keys used for spiral neighborhood iteration int raster_scan_order_8[8] = {0, 1, 2, 3, 5, 6, 7, 8};
int spiral_scan_order_8[8] = {1, 2, 5, 8, 7, 6, 3, 0};
int spiral_scan_order_24[24] = {7, 8, 13, 18, 17, 16, 11, 6, 1, 2,
3, 4, 9, 14, 19, 24, 23, 22, 21, 20,
15, 10, 5, 0};
int spiral_scan_order_48[48] = {17, 18, 25, 32, 31, 30, 23, 16, 9,
0, 11, 12, 19, 26, 37, 40, 39, 38, 37,
36, 29, 22,
15, 8, 1, 2, 3, 4, 5, 6, 13, 20, 27,
34, 41, 48, 47, 46, 45, 44, 43, 42,
35, 28, 21, 14, 7, 0};

// Constant GPGPU memory allocations

  constant  
  constant  

double deviceConstKernel[3*3]; int deviceScanOrder[48];



  global  

void transpose_kernel(double *result, double *input,int rows, int cols) 
{

    int row = blockIdx.y * blockDim.y + threadIdx.y; int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < rows && col < cols) {
    result[row * cols + col] = input[col * rows + row];
    }
}



  global  

cols) {

void array_multiply_kernal(double *a, double *b,
double *result, int rows, int

int ty = blockIdx.y * blockDim.y + threadIdx.y; int tx = blockIdx.x * blockDim.x + threadIdx.x;

if(ty < rows && tx < cols) {
result[ty * cols + tx] = a[ty * cols + tx] * b[ty * cols + tx];
}
}

  global  

void convolve_kernel_constant(unsigned char *image, double *result, int rows, int cols) {

    int ty = blockIdx.y * blockDim.y + threadIdx.y; int tx = blockIdx.x * blockDim.x + threadIdx.x; int kernel_offset = 3.0f/ 2.0f;
    int image_row = ty + kernel_offset; int image_col = tx + kernel_offset;

    if(image_row < rows - kernel_offset && image_col < cols - kernel_offset) {


    double value = 0.0f; 
    for(int i=0; i<3; ++i) {
        int row = (image_row - kernel_offset) + i; 
        for(int j=0; j<3; ++j) {
            int col = (image_col - kernel_offset) + j; value += deviceConstKernel[i * 3 + j] *
            (double)image[row * cols + col];
            }
        }
        result[image_row * cols + image_col] = value;
    }
}

 template<typename T>
  global void convolve_kernel_seperable_vertical(T *image,
double *result, int rows, int cols, double a, double b, double c) 
{ 
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x; int kernel_offset = 3.0f/ 2.0f;
    int image_row = ty; int image_col = tx;

    if(image_row < rows - kernel_offset && image_col < cols - kernel_offset && image_row >= kernel_offset && image_col >= kernel_offset) {

    result[image_row * cols + image_col] = a * image[(image_row-1)*cols + image_col] +
                                        b * image[image_row * cols + image_col] +
                                        c * image[(image_row +1) * cols + image_col];
    }
}

 template<typename T>
  global void convolve_kernel_seperable_horizontal(T *image,double *result, int rows, int cols, double a, double b, double c) {
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x; int kernel_offset = 3.0f/ 2.0f;
    int image_row = ty; int image_col = tx;

    if(image_row < rows - kernel_offset && image_col < cols - kernel_offset && image_row >= kernel_offset && image_col >= kernel_offset) {

    result[image_row * cols + image_col] = a * image[image_row*cols + image_col - 1] +
                                        b * image[image_row * cols + image_col] +
                                        c * image[image_row * cols + image_col + 1];
    }
}


  global void convolve_kernel_seperable_horizontal_row(unsigned char *image, double *result, int rows, int cols, int row, double a, double b, double c) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x; int kernel_offset = 3.0f/ 2.0f;
    int image_col = tx + kernel_offset;

    if(image_col < cols - kernel_offset) { result[row * cols + image_col] = a *
    image[row * cols + image_col - 1] +
                                b * image[row * cols + image_col] +
                                c * image[row * cols + image_col + 1];
    }
}


  device  

static double sum_neighbors(double *image, int row, int col,int cols, int window_dim) {

    int window_center = window_dim / 2.0f; double sum = 0.0f;
    for(int i=0; i<window_dim; ++i) { 
        for(int j=0; j<window_dim; ++j) {
            int image_row = (row - window_center) + i; int image_col = (col - window_center) + j;

            sum += image[image_row * cols + image_col];
        }
    }
    return sum;
}



  device  

static double sum_neighbors_integral(double *image, int row, int col, int cols, int window_dim) {

    int win_off = window_dim / 2.0f;

    double a = image[(row - win_off - 1) * cols + (col - win_off - 1)]; double b = image[(row - win_off - 1) * cols + (col + win_off)]; double c = image[(row + win_off ) * cols + (col - win_off - 1)]; double d = image[(row + win_off) * cols + (col + win_off)];

    return a + d - b - c;
    }



    host device  

    static void eigen_values(double M[2][2],
    double *l1, double *l2) {

    double d = M[0][0];
    double e = M[0][1];
    double f = M[1][0];
    double g = M[1][1];

    *l1 = ((d + g) + sqrt(pow(d + g, 2.0) - 4*(d*g - f*e))) / 2.0f;
    *l2 = ((d + g) - sqrt(pow(d + g, 2.0) - 4*(d*g - f*e))) / 2.0f;
}



static global  
 
void detect_corners_kernel(double *dx2, double *dy2, double *dydx, int rows, int cols, double k,
double *corner_response, int window_dim) {

    int ty = blockIdx.y * blockDim.y + threadIdx.y; int tx = blockIdx.x * blockDim.x + threadIdx.x; int window_offset = window_dim / 2.0f;
    int image_row = ty; int image_col = tx; double M[2][2];

    if(image_row < rows - window_offset && image_col < cols - window_offset && image_row >= window_offset && image_col >= window_offset) {

    M[0][0] = sum_neighbors(dx2, image_row, image_col,
    cols, window_dim);
    M[0][1] = sum_neighbors(dydx, image_row, image_col,
                                cols,  window_dim); M[1][1] = sum_neighbors(dy2, image_row, image_col,
    cols, window_dim);
    M[1][0] = M[0][1];

    double l1, l2; eigen_values(M, &l1, &l2);

    double r = l1 * l2 - k * pow(l1 + l2, 2.0); corner_response[image_row * cols + image_col] = r > 0? r : 0;
    }
}


  global void detect_corners_integral_kernel(double *dx2,
double *dy2, double *dydx, int rows, int cols, double k, double *corner_response, int window_dim)
 {
    int ty = blockIdx.y * blockDim.y + threadIdx.y; int tx = blockIdx.x * blockDim.x + threadIdx.x; int window_offset = window_dim / 2.0f;
    int image_row = ty; int image_col = tx; double M[2][2];

    if(image_row < rows - window_offset && image_col < cols - window_offset && image_row >= window_offset && image_col >= window_offset) {

        M[0][0] = sum_neighbors_integral(dx2, image_row,
                                                image_col, cols, window_dim); M[0][1] = sum_neighbors_integral(dydx, image_row,
                                                image_col, cols, window_dim); M[1][1] = sum_neighbors_integral(dy2, image_row,
        image_col, cols, window_dim);
        M[1][0] = M[0][1];

        double l1 = 6; double l2 = 7;
        eigen_values(M, &l1, &l2);

        double r = l1 * l2 - k * pow(l1 + l2, 2.0); corner_response[image_row * cols + image_col] = r > 0 ? r : 0;
    }
}

template <typename T>
static double *convolve_seperable(T *devInput, double *devResult, int rows, int cols, double rx, double ry, double rz,
double vx, double vy, double vz) {

dim3 dimGrid(ceil(cols/ (double)TILE_DIM),
                ceil(rows/ (double)TILE_DIM)); dim3 dimBlock(TILE_DIM, TILE_DIM);

convolve_kernel_seperable_horizontal<T> <<< dimGrid, dimBlock
        >>>(devInput, deviceResultTemp, rows, cols, rx, ry, rz); CUDA_SAFE(cudaDeviceSynchronize()); convolve_kernel_seperable_vertical<double> <<< dimGrid,

dimBlock vz);

}


        >>>(deviceResultTemp, devResult, rows, cols, vx, vy, return devResult;

  global  

{


void non_maxima_suppression_pattern_kernel(double *image, double *result, int rows, int cols, int pattern_size)


int ty = blockIdx.y * blockDim.y + threadIdx.y; int tx = blockIdx.x * blockDim.x + threadIdx.x; int row = ty;
int col = tx;

int DIM = sqrt((double)pattern_size + 1); int OFF = DIM / 2;

if(row >= OFF && row < rows - OFF && col >= OFF && col < cols - OFF) {
double pixel = image[row * cols + col]; for(int i=0; i < pattern_size; ++i) {
int pr = deviceScanOrder[i] / DIM;

int pc = deviceScanOrder[i] % DIM;

int ir = (row - OFF) + pr; int ic = (col - OFF) + pc;

if(image[ir * cols + ic] > pixel) { pixel = 0;
break;
}
}
result[row * cols + col] = pixel;
}
}

static void array_multiply(double *devA, double *devB, double
*devResult,
int rows, int cols) {
     dim3 dimGrid(ceil(cols/ (double)TILE_DIM), ceil(rows/ (double)TILE_DIM));
dim3 dimBlock(TILE_DIM, TILE_DIM);

array_multiply_kernal<<< dimGrid, dimBlock
          >>>(devA, devB, devResult, rows, cols); CUDA_SAFE(cudaDeviceSynchronize());
}

static void corner_detector(double *devDx2, double *devDy2,
double *devDxDy, double *devCornerResponse, int rows, int cols, double k, int window_dim) {
dim3 dimGrid(ceil(cols/ (double)TILE_DIM),
                ceil(rows / (double)TILE_DIM)); dim3 dimBlock(TILE_DIM, TILE_DIM); detect_corners_kernel <<< dimGrid, dimBlock
>>> (devDx2, devDy2, devDxDy,
rows, cols, k, devCornerResponse, window_dim);

CUDA_SAFE(cudaDeviceSynchronize());

}


static void corner_detector_integral(double *devDx2, double *devDy2, double *devDxDy, double *devCornerResponse, int rows, int cols, double k, int window_dim) {
dim3 dimGrid(ceil(cols/ (double)TILE_DIM),
                ceil(rows / (double)TILE_DIM)); dim3 dimBlock(TILE_DIM, TILE_DIM);

 detect_corners_integral_kernel <<< dimGrid, dimBlock >>> (devDx2, devDy2, devDxDy, rows, cols, k, devCornerResponse,
window_dim); CUDA_SAFE(cudaDeviceSynchronize());
}


static void inclusive_scan_rows(double *devResult, double *devInput,
                                int rows, int cols, int *keys) { thrust::device_ptr<double> input =
thrust::device_pointer_cast(devInput); thrust::device_ptr<double> output =
thrust::device_pointer_cast(devResult);
thrust::device_ptr<int> k = thrust::device_pointer_cast(keys); thrust::exclusive_scan_by_key(k, k + rows * cols, input, output);
}

static void integral_image(double *devResult, double *devInput,
                          int rows, int cols) { dim3 dimBlock(TILE_DIM, TILE_DIM);
double *devRotated =	deviceResultTemp; inclusive_scan_rows(devResult, devInput, rows, cols, scanKeys);
dim3 dimGridTranspose(ceil(rows/ (double)TILE_DIM),
                          ceil(cols/ (double)TILE_DIM)); transpose_kernel <<< dimGridTranspose, dimBlock
          >>> (devRotated, devResult, cols, rows); CUDA_SAFE(cudaDeviceSynchronize());
inclusive_scan_rows(devRotated, devRotated, cols, rows, scanKeysT); dim3 dimGrid(ceil(cols/ (double)TILE_DIM), ceil(rows/
(double)TILE_DIM));
transpose_kernel <<< dimGrid, dimBlock >>> (devResult, devRotated,
rows, cols);
CUDA_SAFE(cudaDeviceSynchronize());
}


static void non_maxima_suppression(double *devResult, double *devInput, int rows, int cols, int *access_pattern, int pattern_size) {
dim3 dimGrid(ceil(cols/ (double)TILE_DIM),
                ceil(rows/ (double)TILE_DIM)); dim3 dimBlock(TILE_DIM, TILE_DIM);

cudaMemcpyToSymbol(deviceScanOrder, access_pattern,
                      pattern_size * sizeof(int)); non_maxima_suppression_pattern_kernel <<< dimGrid, dimBlock
          >>> (devInput, devResult, rows, cols, pattern_size); CUDA_SAFE(cudaDeviceSynchronize());
}

namespace harris_detection { namespace optimized {

void detect_features(std::vector<cv::KeyPoint> &features, unsigned char *image, int rows, int cols, double k, double thresh, int window_dim) {
double *deviceSmoothed = deviceResult[0]; double *deviceDx = deviceResult[1]; double *deviceDy = deviceResult[2]; double *deviceDxDy = deviceResult[3];
double *deviceDx2Integral = deviceResult[4]; double *deviceDy2Integral = deviceResult[5]; double *deviceDxDyIntegral = deviceResult[7]; double *deviceCornerResponse = deviceResult[7];

cudaMemcpy(deviceImage, image, rows * cols, cudaMemcpyHostToDevice);

convolve_seperable<unsigned char>(deviceImage, deviceSmoothed,
            rows, cols, 1/16.0f, 2/16.0f, 1/16.0f, 1, 2, 1); CUDA_SAFE(cudaDeviceSynchronize());

convolve_seperable<double>(deviceSmoothed, deviceDx, rows, cols, -1, 0, 1, 1, 2, 1);
CUDA_SAFE(cudaDeviceSynchronize());

convolve_seperable<double>(deviceSmoothed, deviceDy, rows, cols, 1, 2, 1, -1, 0, 1);
CUDA_SAFE(cudaDeviceSynchronize());

array_multiply(deviceDx, deviceDy, deviceDxDy, rows, cols); array_multiply(deviceDx, deviceDx, deviceDx, rows, cols); array_multiply(deviceDy, deviceDy, deviceDy, rows, cols);

corner_detector(deviceDx, deviceDy, deviceDxDy,
deviceCornerResponse, rows, cols, k, window_dim);
double *deviceSuppressedCornerResponse = deviceResult[0];

non_maxima_suppression(deviceSuppressedCornerResponse, deviceCornerResponse, rows, cols, spiral_scan_order_8, 8);

double *hostSuppressedCornerResponse = to_host<double>(deviceSuppressedCornerResponse,
                          rows, cols); for(int i=0; i < rows; i++) {

for(int j=0; j < cols; ++j) {
     if(hostSuppressedCornerResponse[i * cols + j]> 0.0) {
        features.push_back(cv::KeyPoint(j, i, 5, -1));
        }
        }
    }

}


void initialize_streams(int count) {
    deviceStreamCount = count;
    deviceStreams = new cudaStream_t[deviceStreamCount];
    for(int i=0; i<deviceStreamCount; ++i) {
        cudaStreamCreate(&deviceStreams[i]);
    }
}

void initialize_image(int rows, int cols) {
    deviceImage = alloc_device<unsigned char>(rows, cols); deviceResultTemp = alloc_device<double>(rows, cols, true); int *hscanKeys = new int[rows * cols];
    int *hscanKeysT = new int[rows * cols];

    for(int i=0; i < rows; ++i) { for(int j=0; j < cols; ++j) {
        hscanKeys[i * cols + j] = i;
        }
    }

    int trows = cols; int tcols = rows;

    for(int i=0; i < trows; ++i) { for(int j=0; j < tcols; ++j) {
        hscanKeysT[i * tcols + j] = i;
        }
    }

    scanKeys = to_device<int>(hscanKeys, rows, cols); scanKeysT = to_device<int>(hscanKeysT, rows, cols);

    delete hscanKeys; delete hscanKeysT;

    for(int i=0; i<DEVICE_RESULT_COUNT; ++i)
    deviceResult[i] = alloc_device<double>(rows, cols, true);
}

void initialize_kernel(int rows, int cols) { deviceKernel = alloc_device<double>(rows, cols);
}

void clean_up() { 
    if(deviceKernel) {
        cudaFree(deviceKernel); 
        deviceKernel = NULL;
    }
    if(deviceImage) {
         cudaFree(deviceImage); deviceImage = NULL;
    }

    for(int i=0; i<DEVICE_RESULT_COUNT; ++i) {
        cudaFree(deviceResult[i]);
    }

        cudaFree(deviceResultTemp); cudaFree(scanKeys); cudaFree(scanKeysT);
    }
}
}



