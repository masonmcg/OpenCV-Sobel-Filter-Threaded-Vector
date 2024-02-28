#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

#define RED 0.2126f
#define GREEN 0.7152f
#define BLUE 0.0722f

#define ESC 27

#define GX {1, 0, -1, 2, -2, 1, 0, -1} //{1, 0, -1, 2, 0, -2, 1, 0, -1}
#define GY {1, 2, 1, 0, 0, -1, -2, -1}//{1, 2, 1, 0, 0, 0, -1, -2, -1}

struct ThreadData
{
    cv::Mat* input;
    cv::Mat* output;
};

void to442_grayscale(cv::Mat *rgbImage);
void to442_sobel(ThreadData *data);

cv::Mat frame_split_stitch(cv::Mat& frame);
void frame_to_sobel(ThreadData *data);

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file>" << std::endl;
        return 1;
    }
    
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
		std::cerr << "Error: Couldn't open the video file." << std::endl;
        return 1;
	}
	
	while (true) {
		
		cv::Mat frame;
		cap >> frame;
		if (frame.empty())
			break;
		
		cv::Mat sobelFiltered = frame_split_stitch(frame);
		
		cv::imshow("Processed Frame", sobelFiltered);
		
		int key = cv::waitKey(30);
		if (key == ESC)
			break;
	}
	
	cap.release();
	cv::destroyAllWindows();
}

cv::Mat frame_split_stitch(cv::Mat& frame)
{
	cv::Size sz = frame.size();
    int cols = sz.width;
    int rows = sz.height;
    int midRow = rows / 2;
    int midCol = cols / 2;
    
    cv::Rect quad1(0, 0, midCol + 1, midRow + 1);
    cv::Rect quad2(midCol, 0, midCol, midRow + 1);
    cv::Rect quad3(0, midRow, midCol + 1, midRow);
    cv::Rect quad4(midCol, midRow, midCol, midRow);
    
    cv::Mat quad1Mat = frame(quad1);
    cv::Mat quad2Mat = frame(quad2);
    cv::Mat quad3Mat = frame(quad3);
    cv::Mat quad4Mat = frame(quad4);
    
    cv::Mat sobelFiltered1(quad1Mat.size().height-2, quad1Mat.size().width-2, CV_8UC3);
    cv::Mat sobelFiltered2(quad2Mat.size().height-2, quad2Mat.size().width-2, CV_8UC3);
    cv::Mat sobelFiltered3(quad3Mat.size().height-2, quad3Mat.size().width-2, CV_8UC3);
    cv::Mat sobelFiltered4(quad4Mat.size().height-2, quad4Mat.size().width-2, CV_8UC3);
    
    pthread_t frame_to_sobel_1_thread, frame_to_sobel_2_thread, frame_to_sobel_3_thread, frame_to_sobel_4_thread;
    
    ThreadData data1 = { &quad1Mat, &sobelFiltered1 };
    ThreadData data2 = { &quad2Mat, &sobelFiltered2 };
    ThreadData data3 = { &quad3Mat, &sobelFiltered3 };
    ThreadData data4 = { &quad4Mat, &sobelFiltered4 };
    
    pthread_create(&frame_to_sobel_1_thread, NULL, (void* (*)(void*)) frame_to_sobel, &data1);
    pthread_create(&frame_to_sobel_2_thread, NULL, (void* (*)(void*)) frame_to_sobel, &data2);
    pthread_create(&frame_to_sobel_3_thread, NULL, (void* (*)(void*)) frame_to_sobel, &data3);
    pthread_create(&frame_to_sobel_4_thread, NULL, (void* (*)(void*)) frame_to_sobel, &data4);
    
    pthread_join(frame_to_sobel_1_thread, NULL);
    pthread_join(frame_to_sobel_2_thread, NULL);
    pthread_join(frame_to_sobel_3_thread, NULL);
    pthread_join(frame_to_sobel_4_thread, NULL);
    
    cv::Mat sobelFiltered, sobelFilteredTop, sobelFilteredBottom;
    
    cv::hconcat(sobelFiltered1, sobelFiltered2, sobelFilteredTop);
    cv::hconcat(sobelFiltered3, sobelFiltered4, sobelFilteredBottom);
    cv::vconcat(sobelFilteredTop, sobelFilteredBottom, sobelFiltered);
    
    return sobelFiltered;
	}

// just a placeholder, edit this later to combine the 2 functions
void frame_to_sobel(ThreadData *data)
{
	to442_grayscale(data->input);
	to442_sobel(data);
}

void to442_grayscale(cv::Mat *rgbImage)
{	
    cv::Size sz = rgbImage->size();
    int imageWidth = sz.width;
    int imageHeight = sz.height;
    
    // Convert float weights to fixed-point representation
    uint16_t blue_weight = (uint16_t)(BLUE * 256);
    uint16_t green_weight = (uint16_t)(GREEN * 256);
    uint16_t red_weight = (uint16_t)(RED * 256);
    
    int i = 0; // height (row) index
    int j = 0; // width (column) index

    for (i = 0; i < imageHeight; i++)
    {
        for (j = 0; j < imageWidth; j+=8)
        {
			if (j + 8 <= imageWidth)
			{
				cv::Vec3b* pixel = &rgbImage->at<cv::Vec3b>(i, j); // Get a pointer to the pixel
				
				// Load four pixels into NEON vectors
				uint8x8x3_t pixels = vld3_u8((uint8_t*)pixel);
				uint8x8_t blue = pixels.val[0];
				uint8x8_t green = pixels.val[1];
				uint8x8_t red = pixels.val[2];
				
				// Convert RGB to grayscale using NEON vectors
				uint16x8_t blue_scaled = vmull_u8(blue, vdup_n_u8(blue_weight));
				uint16x8_t green_scaled = vmull_u8(green, vdup_n_u8(green_weight));
				uint16x8_t red_scaled = vmull_u8(red, vdup_n_u8(red_weight));
				uint16x8_t gray = vaddq_u16(blue_scaled, vaddq_u16(green_scaled, red_scaled));
				
				// Shift right by 8 bits to divide by 256 (equivalent to multiplying by 1/256)
				gray = vshrq_n_u16(gray, 8);
				
				// Pack the 16-bit grayscale values to 8-bit
				uint8x8_t gray_u8 = vqmovn_u16(gray);
				
				pixels.val[0] = gray_u8;
				
				// Store the grayscale values back to memory
				vst3_u8((uint8_t*)pixel, pixels);
			}
            else
            {
				cv::Vec3b& pixel = rgbImage->at<cv::Vec3b>(j, i); // Vec<uchar, 3>
				uchar blue = pixel[0];
				uchar green = pixel[1];
				uchar red = pixel[2];
				
				blue = blue * BLUE;
				green = green * GREEN;
				red = red * RED;
				
				uchar gray = blue + green + red;

				pixel[0] = gray;
				
				j -= 7;
			}
        }
    }
}


void to442_sobel(ThreadData *data) 
{
    cv::Size sz = data->input->size();
    int imageWidth = sz.width;
    int imageHeight = sz.height;
    
    int16_t gx[] = GX;
    int16_t gy[] = GY;
    int16_t grayValues[8];

    int i = 0; // width (column) index
    int j = 0; // height (row) index
    
    for (i = 1; i < imageWidth - 1; i++)
    {
        for (j = 1; j < imageHeight - 1; j++)
        {
            int index = 0;
            for (int dj = -1; dj <= 1; ++dj)
            {
                for (int di = -1; di <= 1; ++di)
                {
					if (!(di == 0 && dj == 0))
					{
						int nj = j + dj;
						int ni = i + di;

						grayValues[index] = data->input->at<cv::Vec3b>(nj, ni)[0];
						index++;
					}
                }      
            }
            
            int16x8_t grayValues_vector = vld1q_s16(grayValues);
            
            int16x8_t gx_sum = vmulq_s16(vld1q_s16(gx), grayValues_vector);
            int16x8_t gy_sum = vmulq_s16(vld1q_s16(gy), grayValues_vector);
            
			int16x8_t sum_vector = vaddq_s16(gx_sum, gy_sum);
			
			int sum = abs((int) vaddvq_s16(sum_vector));
            
			// Clamp sum value to 255
            sum = std::min(sum, 255);
            
            data->output->at<cv::Vec3b>(j - 1, i - 1) = cv::Vec3b(sum, sum, sum);
        }
    }
}
