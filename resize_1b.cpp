#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

#define IMAGE_RAW_ROW  360
#define IMAGE_RAW_COL  640
#define IMAGE_ROW  160
#define IMAGE_COL  320

// void resizeNearestNeighbor(const uint8_t* input, uint8_t* output, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
// {
//     const int x_ratio = (int)((sourceWidth << 16) / targetWidth);
//     const int y_ratio = (int)((sourceHeight << 16) / targetHeight) ;
//     const int colors = 3;

//     for (int y = 0; y < targetHeight; y++)
//     {
//         int y2_xsource = ((y * y_ratio) >> 16) * sourceWidth;
//         int i_xdest = y * targetWidth;

//         for (int x = 0; x < targetWidth; x++)
//         {
//             int x2 = ((x * x_ratio) >> 16) ;
//             int y2_x2_colors = (y2_xsource + x2) * colors;
//             int i_x_colors = (i_xdest + x) * colors;

//             output[i_x_colors]     = input[y2_x2_colors];
//             output[i_x_colors + 1] = input[y2_x2_colors + 1];
//             output[i_x_colors + 2] = input[y2_x2_colors + 2];
//         }
//     }
// }

// void resizeNearestNeighbor(const uint8_t* input, uint8_t* output, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
// {
//     const int colors = 3;

//     int x_map[IMAGE_COL];
//     int y_map[IMAGE_ROW];

//     const int x_ratio = (int)((sourceWidth << 16) / targetWidth);
//     const int y_ratio = (int)((sourceHeight << 16) / targetHeight);

//     for (int x = 0; x < targetWidth; x++) {
//         x_map[x] = ((x * x_ratio) >> 16) * colors;
//     }

//     for (int y = 0; y < targetHeight; y++) {
//         y_map[y] = ((y * y_ratio) >> 16) * sourceWidth * colors;
//     }

//     for (int y = 0; y < targetHeight; y++) {
//         int i_xdest = y * targetWidth * colors;
//         int input_row_offset = y_map[y];

//         for (int x = 0; x < targetWidth; x++) {
//             int input_offset  = input_row_offset + x_map[x];
//             int output_offset = i_xdest + x * colors;

//             output[output_offset]     = input[input_offset];
//             output[output_offset + 1] = input[input_offset + 1];
//             output[output_offset + 2] = input[input_offset + 2];
//         }
//     }
// }

void resizeNearestNeighbor(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                           int sourceWidth, int sourceHeight,
                           int targetWidth, int targetHeight)
{
    const int x_ratio = (int)((sourceWidth << 16) / targetWidth);
    const int y_ratio = (int)((sourceHeight << 16) / targetHeight);
    const int colors = 3;

    int* x2_table = (int*)malloc(targetWidth * sizeof(int));
    for (int x = 0; x < targetWidth; x++) {
        x2_table[x] = ((x * x_ratio) >> 16) * colors;
    }

    for (int y = 0; y < targetHeight; y++) {
        int y2 = (y * y_ratio) >> 16;
        int y2_base = y2 * sourceWidth * colors;
        int out_row = y * targetWidth * colors;

        for (int x = 0; x < targetWidth; x++) {
            int in_index = y2_base + x2_table[x];
            int out_index = out_row + x * colors;

            output[out_index]     = input[in_index];
            output[out_index + 1] = input[in_index + 1];
            output[out_index + 2] = input[in_index + 2];
        }
    }

    free(x2_table);
}

           
extern "C" {
    void load_image(uint8_t *src, uint8_t *dst) {
        resizeNearestNeighbor(src,dst,IMAGE_RAW_COL,IMAGE_RAW_ROW,IMAGE_COL,IMAGE_ROW);
    }
    
}
