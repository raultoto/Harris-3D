
#include "harris_detector_cpu.h"

using namespace std; using namespace cv;
#define MIN(a, b) ((a) < (b) ? a : b) namespace harris_detection {

template<typename T>
static double *convolve(T *image, unsigned image_rows, unsigned image_cols, double *kernal, int kernal_dim) {
    unsigned kernal_center = kernal_dim / 2.0f;
    double *output = new double[image_rows * image_cols];

    for(int i=kernal_center; i < image_rows - kernal_center; ++i) { for(int j=kernal_center; j < image_cols - kernal_center;
        ++j) {
            double sum = 0.0f;

            for(int k=0; k < kernal_dim; ++k) {
                unsigned image_row = (i - kernal_center) + k; for(int v=0; v < kernal_dim; ++v) {
                unsigned image_col = (j - kernal_center) + v;

                sum += kernal[k * kernal_dim + v] * image[image_row * image_cols + image_col];
                }
            }
            output[i * image_cols + j] = sum;
        }
    }
    return output;
}


static void double_to_image(unsigned char *dst, double *src,int rows, int cols) {
     for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            dst[i * cols + j] = (unsigned char)src[i * cols + j];
        }
    }
}



static double *array_multiply(double *a, double *b, int rows, int cols) { double *product = new double[rows * cols];

    for(int i=0; i<rows; ++i) {
         for(int j=0; j<cols; ++j) {
            product[i * cols + j] = a[i * cols + j] *
            b[i * cols +j];
        }
    }
    return product;
}

static double sum_neighbors(double *image, int row, int col,int cols, int window_dim) {
    int window_center = window_dim / 2.0f;
    double sum = 0.0f;
    for(int i=0; i<window_dim; ++i) { 
        for(int j=0; j<window_dim; ++j) {
            int image_row = (row - window_center) + i; int image_col = (col - window_center) + j;

            sum += image[image_row * cols + image_col];
        }
    }
    return sum;
}


static void eigen_values(double M[2][2], double &l1, double &l2) { double d = M[0][0];
double e = M[0][1];
double f = M[1][0];
double g = M[1][1];

l1 = ((d + g) + sqrt(pow(d + g, 2.0f) - 4*(d*g - f*e))) / 2.0f;
l2 = ((d + g) - sqrt(pow(d + g, 2.0f) - 4*(d*g - f*e))) / 2.0f;
}

static void linear_scale(double *data, int rows, int cols,double new_min, double new_max) { 
     double old_min = *std::min_element(data, data + rows * cols); double old_max = *std::max_element(data, data + rows * cols);

    for(int i=0; i<rows; ++i) { for(int j=0; j<cols; ++j) {
        data[i * cols + j] =	MIN(10 *
        (((new_max - new_min) * (data[i * cols + j]) / (old_max - old_min)) + new_min), 255);

        }
    }
}


static void draw_circles(Mat &rgb, double *corner_response,int rows, int cols) {

    for(int i=0; i<rows; ++i) { 
        for(int j=0; j<cols; ++j) {
            if(corner_response[i * cols + j] > 0.0f) { 
                cv::circle(rgb, Point(j, i), 5,
                cv::Scalar(0, 0, 255), 2);
            }
        }
    }
}

static void non_maxima_suppression_raster(double *input, double *output, int rows, int cols, int win_dim) {
unsigned win_center = win_dim / 2.0f; bool running;

for(int i=win_center; i < rows - win_center; ++i) 
{ 
    for(int j=win_center; j < cols - win_center; ++j) {
    double pixel = input[i * cols + j];

    running = true;
    for(int k=0; running && k < win_dim; ++k) { 
        for(int v=0; running && v < win_dim; ++v) {
            unsigned image_row = (i - win_center) + k; unsigned image_col = (j - win_center) + v;

// Don't count the center pixel
            if(k == win_center && v == win_center) continue;

            if(pixel < input[image_row * cols + image_col])
    {
        pixel = 0; running = false;
    }
    }
}
    output[i * cols + j] = pixel;
}
}
}

void detect_features(std::vector<cv::KeyPoint> &features,unsigned char *image, int rows, int cols, double k, double thresh, int nms_dim) {

    // De-noise input image
    double *smoothed = convolve(image, rows, cols, gaussian_3x3,
    KERNAL_DIM);

    // Determine x and y gradients
    double *dx = convolve(smoothed, rows, cols, sobel_x,
    KERNAL_DIM);

    double *dy = convolve(smoothed, rows, cols, sobel_y,
    KERNAL_DIM);

    // Square gradients for harris matrix calculation double *dx2 = array_multiply(dx, dx, rows, cols); double *dxdy = array_multiply(dx, dy, rows, cols); double *dy2 = array_multiply(dy, dy, rows, cols);

    int window_center = WINDOW_DIM / 2.0f; double M[2][2] = {0.0f};

    double *corner_response = new double[rows * cols]();

    // Iterate over squared gradients and compute harris matrix R scores for(int i=window_center; i<rows - window_center; ++i) {
    for(int j=window_center; j<cols - window_center; ++j) { 
        M[0][0] = sum_neighbors(dx2, i, j, cols, WINDOW_DIM); M[0][1] = sum_neighbors(dxdy, i, j, cols, WINDOW_DIM); M[1][0] = M[0][1];
        M[1][1] = sum_neighbors(dy2, i, j, cols, WINDOW_DIM); double l1, l2;
        eigen_values(M, l1, l2);

        double R = l1 * l2 - k * pow(l1 + l2, 2.0f);

        // Threshold R score

        if(R > thresh) {
            corner_response[i * cols + j] = R;
        }
    }
}

double *suppressed = new double[rows * cols](); non_maxima_suppression_raster(corner_response, suppressed,
rows, cols, nms_dim);

        for(int i=0; i < rows; i++) { 
            for(int j=0; j < cols; ++j) {
                if(suppressed[i * cols + j] > 0.0) { 
                    features.push_back(cv::KeyPoint(j, i, 5, -1));
                }
            }
        }
    }
}