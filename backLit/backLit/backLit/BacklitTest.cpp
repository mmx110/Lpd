// BacklightDeal.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>
#include <opencv2\opencv.hpp>
#include "backLit.h"

using namespace std;
using namespace cv;

//cv::Rect &operator *(cv::Rect &rOne, int scale)
//{
//	rOne.x*=scale;
//	rOne.y*=scale;
//	rOne+=cv::Size(rOne.width*scale, rOne.height*scale);
//	return rOne;
//}

int _tmain(int argc, _TCHAR* argv[])
{
	fstream in("C:\\Users\\user\\Desktop\\test27\\list.txt"); //change your image file path, you must generate a path list(absulute directory)
	if (!in)
	{
		cout<<"file is not exist"<<endl;
		system("pause");
		return 0;
	}
	string imgName("");
	string windowName = "result";
	int i=0;
	while(getline(in, imgName))
	{
		cout<<imgName<<endl;
		Mat gray, bin, sobelImg;
		Mat img = imread(imgName.c_str());
		Mat reImg, dest;
		resize(img, reImg, Size(img.cols/2, img.rows/2));
		Mat subReImg = reImg.clone();

		vector<Rect> winRect;
		string saveFolder = "F:\\Users\\Monkey\\Desktop\\local\\seg\\";
		unsigned int nums = segImgByExp(img, winRect);
		for (int i=0;i<nums;i++)
		{
			winRect[i].x*=2;
			winRect[i].y*=2;
			winRect[i]+=Size(winRect[i].width, winRect[i].height);
			Rect wid_Rect;									//以宽度为基准，标准化检出结果
			Rect Hei_Rect;
			wid_Rect.width = winRect[i].width;
			wid_Rect.height = saturate_cast<int>(winRect[i].width*7.0/22);
			Hei_Rect.width = saturate_cast<int>(winRect[i].height*22.0/7);
			Hei_Rect.height = winRect[i].height;

			wid_Rect.x = winRect[i].x;
			wid_Rect.y = winRect[i].y-saturate_cast<int>(abs(winRect[i].height - wid_Rect.height)/2);
			Hei_Rect.x = winRect[i].x-saturate_cast<int>(abs(wid_Rect.width-winRect[i].width)/2);
			Hei_Rect.y = winRect[i].y;
			rectangle(img, wid_Rect, Scalar(0,255,0));
		}
		imshow("seg", img);
		waitKey(0);
	}
	system("pause");
	return 0;
}
