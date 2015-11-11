#pragma once

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

int enhanceImgByMAD(const cv::Mat& ssrc, cv::Mat& ddest, cv::Size block);
float calEnchancPixVal(float meanVal, float varianceVal, float pixVal);
void interpolation(const cv::Mat& src, cv::Mat &dest, std::vector<cv::Point> position, std::vector<float> blockMean,std::vector<float> blockVariance, int xNums, int yNums, int xWidth, int yHeight);
void getplateRectbyContoursADDLocalOSTU(cv::Mat image, std::vector<cv::Rect> &possible_plate_rects, float rect_ratio);
int SobelS(cv::Mat image, cv::Mat &dest);
double getThreshVal_Otsu_8u( const cv::Mat& _src );
unsigned int mergeAndDiscardRect(const cv::Mat imgArray[], std::vector<cv::Rect> midRes[], std::vector<cv::Rect> &resVec, int nums);
int mergeRect(std::vector<cv::Rect> &srcRects, float scale);
unsigned int segImgByExp( const cv::Mat colorImg, std::vector<cv::Rect> &realCandidates, cv::Size block = cv::Size(30,30), int iterations = 6);
void verifyCoordinate(cv::Rect &rect, cv::Size igSize);