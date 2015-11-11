#include "stdafx.h"
#include "backLit.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/************************************************************************/
/* 灰度图像增强，使用局部灰度值均值和方差结合双线插值法                  */
/************************************************************************/
int enhanceImgByMAD(const Mat& ssrc, Mat& ddest, Size block)
{
	if (ssrc.empty())
	{
		return -1;
	}
	Mat src, dest;
	ssrc.convertTo(src, CV_32FC1);
	dest = src.clone();
	int imgWidth = src.cols;
	int imgHeight = src.rows;
	int xWidth = block.width;
	int yHeight = block.height;
	vector<Point> location;             //矩形顶点的位置
	vector<float> blockMean;			//矩形顶点的均值
	vector<float> blockVariance;		//矩形顶点的方差
	if (imgWidth/xWidth<2&&imgHeight/yHeight<2)
	{
		return -1;
	}
	if (block.width>=imgWidth||block.height>=imgHeight)
	{
		return -2;
	}
	int xNums = imgWidth/xWidth;
	int yNums = imgHeight/yHeight;
	int xyArea = xWidth*yHeight;
	Mat roiImg(xWidth, yHeight, src.type());
	Point startPoint(0,0);
	for(int i=0;i<yNums;i++)
	{
		startPoint.y=i*yHeight;
		for (int j=0;j<xNums;j++)
		{
			startPoint.x=j*xWidth;
			Rect rect(startPoint.x, startPoint.y, xWidth, yHeight);
			src(rect).copyTo(roiImg);
			float tempSum = 0.0;
			float temMean = 0.0;
			float tempVariance = 0.0;
			Point position(0,0);
			for (Mat_<float>::iterator it=roiImg.begin<float>();it!=roiImg.end<float>();++it)
			{  
				tempSum+=(*it);
			}
			temMean = tempSum/xyArea;
			tempSum = 0.0;
			for (Mat_<float>::iterator it=roiImg.begin<float>();it!=roiImg.end<float>();++it)
			{   
				tempSum+=(*it-temMean)*(*it-temMean);
			}
			tempVariance = std::sqrt(tempSum/xyArea);
			position.x = startPoint.x+xWidth/2;
			position.y = startPoint.y+yHeight/2;
			blockMean.push_back(temMean);
			blockVariance.push_back(tempVariance);
			location.push_back(position);
		}
	}
	if (blockMean.size()>=4&&blockVariance.size()>=4)
	{
		interpolation(src, dest, location, blockMean, blockVariance, xNums, yNums, xWidth, yHeight);
		dest.convertTo(ddest, CV_8UC1);
		return 1;
	}
	else
	{
		return -3;
	}

}
/************************************************************************/
/* 根据公式计算矩形中心点的值                                                 */
/************************************************************************/
float calEnchancPixVal(float meanVal, float varianceVal, float pixVal)
{
	float coefficient = 0.0;
	uchar enhanceVal = 0;
	if (varianceVal<0)
	{
		return 0;
	}
	else if (varianceVal<20.0)
	{
		coefficient = 3/((varianceVal-20)*(varianceVal-20)/200+1);
	}
	else if (varianceVal<60.0&&varianceVal>=20.0)
	{
		coefficient = 3/((varianceVal-20)*(varianceVal-20)/800+1);
	}
	else
	{
		coefficient = 1;
	}
	float enhancePixVal = (pixVal-meanVal)*coefficient+meanVal;
	if (enhancePixVal>=255)
	{
		enhancePixVal = pixVal;
	}
	return enhancePixVal;
}
/************************************************************************/
/* 双线插值                                                                     */
/************************************************************************/
void interpolation(const Mat& src, Mat &dest, vector<Point> position, vector<float> blockMean,vector<float> blockVariance, int xNums, int yNums, int xWidth, int yHeight)
{
	int width = src.cols;
	int height = src.rows;
	int w=dest.cols;
	int h=dest.rows;
	int firPoint = 0;
	int secPoint = 0;
	int thirdPoint = 0;
	int forthPoint = 0;
	for (int i=0;i<yNums-1;i++)
	{
		for (int j=0;j<xNums-1;j++)
		{
			firPoint = i*xNums+j;
			secPoint = (i+1)*xNums+j;
			thirdPoint = firPoint+1;
			forthPoint = secPoint+1;
			for(int n=position[firPoint].y;n<position[forthPoint].y;n++)
			{
				for(int k=position[firPoint].x;k<position[forthPoint].x;k++)
				{
					float cx = 1.0*(k  - xWidth*j-xWidth*0.5)/xWidth;
					float cy = 1.0*(n - yHeight*i-yHeight*0.5)/yHeight;

					float tempMean = (1-cy)*((1-cx)*blockMean[firPoint]+cx*blockMean[thirdPoint])+cy*((1-cx)*blockMean[secPoint]+cx*blockMean[forthPoint]);
					float tempVariance = (1-cy)*((1-cx)*blockVariance[firPoint]+cx*blockVariance[thirdPoint])+cy*((1-cx)*blockVariance[secPoint]+cx*blockVariance[forthPoint]);
					float a = src.at<float>(n, k);
					float b = calEnchancPixVal(tempMean, tempVariance, a);
					dest.at<float>(n, k) = b;
				}
			}
		}
	}
}
unsigned int segImgByExp( const cv::Mat colorImg, std::vector<cv::Rect> &realCandidates, cv::Size block, int iterations)
{
	assert(!colorImg.empty());
	Mat subImg = colorImg.clone();
	resize(subImg, subImg, Size(subImg.cols/2, subImg.rows/2));
	int wid = subImg.cols;
	int hei = subImg.rows;
	double areaThershold = wid*hei/5.0;
	double segAreaThershold = areaThershold/4.0;
	cvtColor(subImg, subImg, CV_RGB2GRAY);
	Mat lit = subImg+50;
	blur(lit, lit, block);
	Mat element = getStructuringElement(MORPH_RECT, block);
	threshold(lit, lit, 0, 255, CV_THRESH_BINARY_INV+CV_THRESH_OTSU);
	morphologyEx(lit, lit, MORPH_OPEN, element);
	Mat *pro = new Mat[iterations];//{Mat::Mat()};
	pro[0] = lit.clone();
	for (int i=0;i<5;i++)
	{
		Mat element = getStructuringElement(MORPH_RECT, block*(i+1));
		morphologyEx(pro[i], pro[i+1], MORPH_OPEN, element);
		/*imshow("bin", pro[i]);
		waitKey(0);*/
	}
	vector<vector<Point>> contours;
	findContours(pro[5], contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<vector<Point>>::iterator pIt = contours.begin();
	int  contIndex = 0;
	while(pIt!=contours.end())
	{
		if (contourArea((Mat)(*pIt))>areaThershold)
		{
			const Point *ppt[1] = {&contours[contIndex][0]};
			int n_size  = (*pIt).size();
			const int *npt = &n_size;
			Mat maskImg = Mat::zeros(Size(subImg.cols, subImg.rows), CV_8UC1);
			fillPoly(maskImg, ppt, npt, 1, Scalar::all(1));						//create mask

			Mat segRes, binSeg, enhImg, edg ;
			segRes.create(subImg.rows, subImg.cols, CV_8UC1);
			multiply(maskImg, subImg, segRes);
			enhanceImgByMAD(segRes, enhImg, Size(11,7));
			SobelS(enhImg, edg);												//get block and image process

			double thVal = getThreshVal_Otsu_8u(edg) - 10;						//get thresh_value
			assert(thVal>0);
			threshold(edg, binSeg, thVal, 255, CV_THRESH_BINARY);
			/*imshow("bin", binSeg);
			waitKey(0);*/
			Mat pro[4];
			Mat ePro[3];
			pro[0] = binSeg.clone();
			vector<vector<Point>> segContour;
			vector<Rect> candidates[3];
			for (int j=0;j<3;j++)
			{
				Mat element = getStructuringElement(MORPH_RECT, Size(9*(j+1),2*(j+1)));		//these parameters maybe adjust
				morphologyEx(pro[j], ePro[j], MORPH_CLOSE, element);
				morphologyEx(ePro[j], pro[j+1], MORPH_ERODE, Mat::ones(2,2,CV_8UC1));		//remove noise, namely small block
				findContours(ePro[j], segContour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
				vector<vector<Point>>::iterator segPit = segContour.begin();
				while(segPit!=segContour.end())
				{
					double courArea = contourArea((Mat)(*segPit));
					if (courArea>100&&courArea<segAreaThershold)
					{
						RotatedRect rotRect = minAreaRect((Mat)(*segPit));
						double wHScale = rotRect.size.width/rotRect.size.height;
						if ((wHScale>0.15&&wHScale<0.5)||(wHScale<9&&wHScale>2))
						{
							Rect tempRect = rotRect.boundingRect();
							if (1.0*tempRect.width/tempRect.height>1.5)
							{
								verifyCoordinate(tempRect, Size(wid, hei));
								candidates[j].push_back(tempRect);
							}
						}
					}
					++segPit;
				}
			}
			mergeAndDiscardRect(ePro, candidates, realCandidates, 3);
			for (int k=0;k<3;k++)
			{
				candidates[k].clear();
			}
		}
		contIndex++;
		++pIt;
	}
	delete[] pro;
	return realCandidates.size();
}

void getplateRectbyContoursADDLocalOSTU(Mat image,vector<Rect> &possible_plate_rects,float rect_ratio)
{
	//imshow(img_name+"原图", change_size(img, 0.4));
	Mat img = image.clone();
	Mat img_copy = img.clone();
	string datainfo; //保存处理过程中的文本信息 最后一次输出
	datainfo.append("方法：图像分割\n");

	vector<Mat> planes;
	Mat img_gray;
	Mat img_gray_light;
	Mat img_gray_black;
	Mat img_gray_light_copy;
	Mat img_gray_black_copy;
	Mat img_gray_copy;
	Mat img_OSTU_threshold;
	Mat img_OSTU_threshold_copy;

	//取HSV空间中的V分量 V = max（R，G，B）
	//cvtColor(img, img, CV_BGR2HSV);
	//split(img, planes);
	img_gray = img.clone();
	img_gray_copy = img.clone();
	
	/**
	把一张图片分割成明暗两个部分
	*/
	threshold(img_gray,img_OSTU_threshold, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	img_OSTU_threshold_copy = img_OSTU_threshold;
	Mat element;
	char i_str[8];
	for (int i = 1; i <= 4; i++)
	{
		element = getStructuringElement(MORPH_RECT,Size(15 * i,15 *i));
		morphologyEx(img_OSTU_threshold,img_OSTU_threshold,MORPH_OPEN,element);
		_itoa(i,i_str,10);
	}
	
	img_gray.copyTo(img_gray_black,img_OSTU_threshold);
	img_gray_light = img_gray - img_gray_black;
	
	img_gray_light_copy = img_gray_light;
	img_gray_black_copy = img_gray_black;
	imshow("light", img_gray_light_copy);
	imshow("black", img_gray_black_copy);
	waitKey(0);
}

int SobelS(Mat image, Mat &dest)
{
	Mat img = image.clone();
	img.convertTo(img,CV_32F);
	Mat res(img.size(),CV_32F,Scalar::all(0));
	int nr = img.rows; //行数
	int nc = img.cols;
	float pixe;
	float min_thresh = 10;
	int delta = 10;
	for (size_t i = delta; i < nr - delta; i++)
	{
		for (size_t j = delta; j < nc - delta; j++)
		{
			//如果方块的左边或者右边全都是0 那么它本身也就是0 避免检测到分块时候出现的边缘
			if ( (img.at<float>(i-1,j-delta) < min_thresh) || (img.at<float>(i,j-delta)  < min_thresh) || (img.at<float>(i+1,j-delta) < min_thresh)  || (img.at<float>(i-1,j+delta) < min_thresh) || (img.at<float>(i,j+delta)   < min_thresh)  || (img.at<float>(i+1,j+delta) < min_thresh) )
			{
				res.at<float>(i,j) = 0;
			}else
			{
				pixe = (img.at<float>(i-1,j-1) +  2 *img.at<float>(i,j-1) + img.at<float>(i+1,j-1)) - ( img.at<float>(i-1,j+1) + 2*img.at<float>(i,j+1) + img.at<float>(i+1,j+1));
				if (pixe >= 0)
				{
					res.at<float>(i,j) =  pixe;
				}else
				{
					res.at<float>(i,j) = -pixe;
				}
			}
		}
	}
	res.convertTo(dest,CV_8UC1);
	return 1;
}
/************************************************************************/
/* function calculate threshold value according OTSU
   code from opecnCv source
*/
/************************************************************************/
double getThreshVal_Otsu_8u( const Mat& _src )
{
	Size size = _src.size();
	if( _src.isContinuous() )
	{
		size.width *= size.height;
		size.height = 1;
	}
	const int N = 256;
	int i, j, h[N] = {0};
	int zeroNum = 0;
	for( i = 0; i < size.height; i++ )
	{
		const uchar* src = _src.data + _src.step*i;
		j = 0;
#if CV_ENABLE_UNROLLED
		for( ; j <= size.width - 4; j += 4 )
		{
			int v0 = src[j];
			h[v0]++;
			int v1 = src[j+1];
			h[v1]++;
			v0 = src[j+2];
			h[v0]++;
			v1 = src[j+3];
			h[v1]++;	
		}
#endif
		for( ; j < size.width; j++ )
		{
			h[src[j]]++;
		}
	}
	double mu = 0, scale = 1./(size.width*size.height);
	/************************************************************************/
	/* 0像素不算在内,otsu的阈值变大，对边缘较弱的车牌有反作用
	/************************************************************************/
	//double mu = 0, scale = 1./(size.width*size.height-h[0]); 
	//h[0] = 0;     //将0像素的统计量置为0
	for( i = 0; i < N; i++ )
		mu += i*(double)h[i];

	mu *= scale;
	double mu1 = 0, q1 = 0;
	double max_sigma = 0, max_val = 0;

	for( i = 0; i < N; i++ )
	{
		double p_i, q2, mu2, sigma;

		p_i = h[i]*scale;
		mu1 *= q1;
		q1 += p_i;
		q2 = 1. - q1;

		if( std::min(q1,q2) < FLT_EPSILON || std::max(q1,q2) > 1. - FLT_EPSILON )
			continue;

		mu1 = (mu1 + i*p_i)/q1;
		mu2 = (mu - q1*mu1)/q2;
		sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
		if( sigma > max_sigma )
		{
			max_sigma = sigma;
			max_val = i;
		}
	}
	return max_val;
}
/************************************************************************/
/* function realize a strategy to merge candidates and discard false candidates
   paramters:
			imgArray:  Array of binary images;
			RectArrat: n-dims array of candidates Rects
			resVec: the final result, A vector contains real candidates of vehicle plates
*/
/************************************************************************/
unsigned int  mergeAndDiscardRect(const Mat imgArray[], vector<Rect> midRes[], vector<Rect> &resVec, int nums)
{
	for (int i=0;i<nums;i++)
	{
		if (imgArray[i].empty())
		{
			return 0;
		}
	}
	for (int i=0; i<nums;i++)
	{
		if (i==nums-1)
		{
			resVec.insert(resVec.begin(), midRes[i].begin(), midRes[i].end());
			break;
		}
		for (vector<Rect>::size_type j=0;j<midRes[i].size();j++)
		{
			for (vector<Rect>::size_type k=0;k<midRes[i+1].size();k++)
			{
				cv::Rect temp = midRes[i][j] & midRes[i+1][k];
				if (temp.area()>midRes[i+1][k].area()*0.6)
				{
					resVec.push_back(temp);
				}
				else if (countNonZero(imgArray[i+1](midRes[i][j]))>temp.area()*0.8)
				{
					resVec.push_back(temp);
				}
			}
		}
	}
	mergeRect(resVec, 0.6);
	return resVec.size();
}

/************************************************************************/
/* function: merge rects
*/
/************************************************************************/
bool sortRule(const Rect &rOne, const Rect &rTwo)
{
	return rOne.x<rTwo.x;
}
int mergeRect(vector<Rect> &srcRects, float scale)
{
	if(srcRects.size()<1)
	{
		return -1;
	}
	std::sort(srcRects.begin(), srcRects.end(), sortRule);
	for (vector<Rect>::size_type i=0;i<srcRects.size()-1;)
	{
		int plusFlag = 0;
		for(vector<Rect>::size_type j=i+1;j<srcRects.size();j++)
		{
			plusFlag = j;
			vector<Rect>::iterator it = srcRects.begin();
			if (srcRects[i].x > srcRects[j].x+srcRects[j].width||srcRects[i].y > srcRects[j].y+srcRects[j].height||srcRects[i].x+srcRects[i].width < srcRects[j].x||srcRects[i].y+srcRects[i].height < srcRects[j].y)
			{	
				i++;
				break;
			}	
			float colInt =  min(srcRects[i].x+srcRects[i].width,srcRects[j].x+srcRects[j].width) - max(srcRects[i].x, srcRects[j].x);
			float rowInt =  min(srcRects[i].y+srcRects[i].height,srcRects[j].y+srcRects[j].height) - max(srcRects[i].y,srcRects[j].y);
			float intersection = colInt * rowInt;
			float area1 = srcRects[i].area();
			float area2 = srcRects[j].area();
			if(intersection/(area1 + area2 - intersection)>scale)
			{
				if (srcRects[i].area()>srcRects[j].area())
				{
					advance(it, j);
				}
				else
				{
					advance(it, i);
				}
				srcRects.erase(it);
				j--;
			}
		}
		if (plusFlag==srcRects.size()-1)
		{
			i++;
		}
	}
	return srcRects.size();
}
/************************************************************************/
/* function: verify cooddinate
*/
/************************************************************************/
void verifyCoordinate(Rect &rect, Size igSize)
{
	int width = igSize.width;
	int height = igSize.height;
	if (rect.x>=width||rect.height>=height)
	{
		rect.x = rect.y = rect.width = rect.height = 0;
	}
	if (rect.x<0)
	{
		rect.x = 0;
	}
	if (rect.y<0)
	{
		rect.y = 0;
	}
	if (rect.width+rect.x>width)
	{
		rect.width = width - rect.x;
	}
	if (rect.height+rect.y>height)
	{
		rect.height = height - rect.y;
	}
}