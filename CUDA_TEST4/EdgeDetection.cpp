

#include "pch.h"
#include <iostream>
#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "cv.h"
#include <process.h>
#include "CameraApi.h"
#include "LaserRange.h"
#include "afxwin.h"
#include "math.h"
#include "cstdlib"
#include "sstream"
#include "ImProcess.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "EdgeDetection.h"
#include "cmath"
#include "omp.h"
#include "Timer.h"
#include "cuda.h"
#include "cuda_runtime.h"

using namespace std;
using namespace cv;

stop_watch watch;

void getPeaker1(Mat matImage, MPoint *point) {
	int Rows = matImage.rows;//y
	//int Cols = matImage.cols;
	int Cols = matImage.cols*matImage.channels();//x
	//int div = 64;
	int MaxPixel;

	/*method 1: the average of the xPisel with same brightness*/
	int Pixeldata;
	int x1 = 0;
	int x2 = 0;
	float step = 0.85;//边沿值百分数
	boolean stepon(false), stepoff(false);
	int getpeak = 0;

	for (int j = 0; j < Rows; j++) {
		uchar* data = matImage.ptr<uchar>(j);
		MaxPixel = data[0];
		for (int i = 1; i < Cols; i++) {
			//data[i] = data[i]; /// div * div + div / 2;
			if (data[i] > MaxPixel) {
				MaxPixel = data[i];
				//point[j].x = i;
			}
		}
		//point[j].y = j;
		point[j].bright = MaxPixel;
		for (int i = 0; i < Cols; i++) {
			Pixeldata = data[i];
			if (stepon == false && Pixeldata > step*MaxPixel) {
				x1 = i;
				stepon = true;
			}
			else if (stepon = true && Pixeldata == MaxPixel && stepoff == false) {
				getpeak++;
			}
			else if (getpeak && Pixeldata < step*MaxPixel) {
				x2 = i - 1;
				stepoff == true;
				break;
			}
		}
		point[j].Pixnum = getpeak;
		point[j].x = (x1 + x2) / 2;
		point[j].y = j;
		point[j].bright = MaxPixel;
		stepon = false;
		stepoff = false;
		getpeak = 0;
		x1 = 0;
		x2 = 0;
		//cout << "(" << point[j].x << "," << point[j].y << "):" << MaxPixel << "    sum:" << point[j].Pixnum << endl;
	}
}

void getPeaker(Mat matImage, int *brightness) {
	int Cols = matImage.cols;
	int Rows = matImage.rows;
	int PixelData;
	int MaxPixel = 0;
	for (int j = 0; j < Rows; j++) {
		uchar* data = matImage.ptr<uchar>(j);
		MaxPixel = data[0];
		for (int i = 1; i < Cols; i++) {
			if (data[i] > MaxPixel) {
				MaxPixel = data[i];
			}
		}
		brightness[j] = MaxPixel;
	}
}

//Canny opencv 边缘检测
void getcanny(Mat matImage, MPoint *point) {
	Mat cloneImage = matImage.clone();
	int g_nCannyLowThreshold = 100;//canny检测低阈值
	Mat tmpImage, dstImage;
	blur(cloneImage, tmpImage, Size(3, 3));
	Canny(tmpImage, dstImage, g_nCannyLowThreshold, g_nCannyLowThreshold * 3);
	namedWindow("canny function");
	imshow("canny function", dstImage);
	int Rows = cloneImage.rows;
	int Cols = cloneImage.cols*cloneImage.channels();
	int x[100];
	int px = 0;
	int PixelDataof;
	int sum = 0;
	double average = 0;
	for (int j = 0; j < Rows; j++) {
		uchar* data = dstImage.ptr<uchar>(j);
		for (int i = 0; i < Cols; i++) {
			PixelDataof = data[i];
			if (PixelDataof > 130) {
				x[px] = i;
				px++;
				sum = sum + i;
				if (px > 100) {
					cout << "there are too many canny points" << endl;
				}
			}
		}
		//逐行计算平均点
		if (px) {

			average = sum * 1.0 / px;
		}
		point[j].cx = average;
		point[j].cy = j;
		average = 0;
		sum = 0;
		px = 0;
		memset(x, 0, px);
		//cout << "(" << point[j].cx << "," << point[j].cy << ")" << endl;
	}
	getErrorIdentifyDouble(cloneImage, point);
	cloneImage.release();
}
//Sobel边缘检测
void getsobel(Mat matImage, MPoint *point) {
	Mat cloneImage = matImage.clone();
	Mat tempImage, tempImage1, tempImage2, tempImage_gauss;
	int ddepth = CV_16S, dx, dy,
		p_ksize = 3,//CV_SCHARR,   //内核 -1相当于3*3内核用schar函数
		p_scale = 2,		//计算导数时缩放因子  何用？？ 调大以后 线条部分更白，背景散点
		p_delta = 0,	//被叠加到倒数的常数 调大了以后 整个背景灰化 更敏锐了？？
		p_borderType = BORDER_DEFAULT;

	//16转8位图  不必修改
	double p_alpha_to8 = 1.0;// 乘数因子
	double p_beta_to8 = 0.0; // 偏移量

	//加权
	double p_alpha_weig_x = 0.9, p_delta_weig_y = 0.1;//xy权重
	double p_gamma_weig = 0; //加权后偏移量
	dx = 1;
	dy = 0;


	//没有处理
	GaussianBlur(cloneImage, tempImage_gauss, Size(7, 7), 0, 0, BORDER_DEFAULT);
	/*namedWindow("gauss");
	imshow("gauss", matImage);*/

	Sobel(tempImage_gauss, tempImage, ddepth, dx, dy, p_ksize, p_scale, p_delta, p_borderType);
	convertScaleAbs(tempImage, tempImage1, p_alpha_to8, p_beta_to8);

	dx = 0;
	dy = 1;
	Sobel(tempImage_gauss, tempImage, ddepth, dx, dy, p_ksize, p_scale, p_delta, p_borderType);
	convertScaleAbs(tempImage, tempImage2, p_alpha_to8, p_beta_to8);

	addWeighted(tempImage1, p_alpha_weig_x, tempImage2, p_delta_weig_y, p_gamma_weig, tempImage, -1);

	namedWindow("sobel");
	imshow("sobel", tempImage);

	//cvtColor(tempImage, tempImage, CV_RGB2GRAY);
	//tempImage为边缘图
	int nr = tempImage.rows;
	int nc = tempImage.cols*tempImage.channels();
	int PixelMin_Range = 80;
	//int PixelMax_Range;
	//int PixelMax=80;
	int n_point = 0;
	int x_sum = 0;
	int Pixel_temp;


	for (int j = 0; j < nr; j++)
	{
		uchar* data = tempImage.ptr<uchar>(j);
		//	PixelMax = data[0];
		for (int i = 0; i < nc; i++)
		{
			Pixel_temp = data[i];
			if (Pixel_temp > PixelMin_Range)
			{
				x_sum = x_sum + i;
				n_point++;
			}
		}
		point[j].cx = x_sum * 1.0 / n_point;
		point[j].cy = j;
		x_sum = 0;
		n_point = 0;
	}
	getErrorIdentifyDoubleW(cloneImage, point, 0.5, 0);
	cloneImage.release();
}

//Laplician 边缘检测
void getlaplacian(Mat matImage, MPoint *point)
{
	Mat cloneImage = matImage.clone();
	Mat tempImage_gauss, tempImage1, tempImage2;
	Mat new_image = Mat::zeros(cloneImage.size(), cloneImage.type());
	int pl_ddepth = CV_16S,
		pl_kernel_size = 3,
		pl_scale = 1,
		pl_delta = 0;
	GaussianBlur(cloneImage, tempImage_gauss, Size(7, 7), 0, 0, BORDER_DEFAULT);
	//namedWindow("gauss");
	//imshow("gauss", tempImage_gauss);


	//cvtColor(tempImage_gauss, tempImage_gauss, CV_RGB2GRAY);
	Laplacian(tempImage_gauss, tempImage1, pl_ddepth, pl_kernel_size, pl_scale, pl_delta, BORDER_DEFAULT);
	convertScaleAbs(tempImage1, tempImage2);

	/*double alpha_co =5;
	double beta_co = 0;
	tempImage2.convertTo(new_image, -1, alpha_co, beta_co);
	tempImage2=new_image.clone();
	new_image.release();

*/

	namedWindow("laplacian");
	imshow("laplacian", tempImage2);

	int nr = tempImage2.rows,
		nc = tempImage2.cols*tempImage2.channels();

	//	cout << nr << " " << nc;
	int PixelMin = 20;
	int x_sum = 0;
	int n_point = 0;
	int max;
	//nr = 1;
	for (int j = 0; j < nr; j++)
	{
		uchar *data = tempImage2.ptr<uchar>(j);
		max = data[0];
		for (int i = 0; i < nc; i++)
		{
			if (data[i] > max)
				max = data[i];
			//cout << (int)data[i] << " ";
			if (data[i] > PixelMin)
			{
				x_sum += i;
				n_point++;
			}
		}
		//cout << max<< endl;
		if (n_point == 0) {
			n_point = 1;
		}
		point[j].cx = x_sum / n_point;
		point[j].cy = j;

		//cout << "(" << point[j].cx << "," << point[j].cy << ")" << endl;
		x_sum = 0;
		n_point = 0;
	}

	getErrorIdentifyDouble(cloneImage, point);
	cloneImage.release();
	//waitKey(0);

}


//亚像素分析函数（待完善）
void getdoublepixel(Mat matImage, MPoint *point) {
	IplImage* image = NULL;
	IplImage* pGrayImage = NULL;
	IplImage* pEdgeImage = NULL;
	CvMat* pDirection = NULL;
	CvMat* pSubEdgeMatH = NULL;
	CvMat* pSubEdgeMatW = NULL;
	CvMat* pGradMat = NULL;
	int nWidth = 0, nHeight = 0;
	CHTimer timer;
	//cvNamedWindow("Orign Image", 1);
	cvNamedWindow("Image", 1);
	image = &IplImage(matImage);
	//cvShowImage("Orign Image", image);
	//Mat OrImage = cvarrToMat(image);
	pGrayImage = cvCreateImage(cvSize(image->width, image->height), image->depth, 1);
	pEdgeImage = cvCreateImage(cvSize(image->width, image->height), image->depth, 1);
	pDirection = cvCreateMat(image->height, image->width, CV_8SC1);
	pSubEdgeMatH = cvCreateMat(image->height, image->width, CV_64FC1);
	pSubEdgeMatW = cvCreateMat(image->height, image->width, CV_64FC1);
	pGradMat = cvCreateMat(image->height, image->width, CV_32SC2);
#if 0
	cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;
	cout << "The channels of the image:" << image->nChannels << endl;
#endif
	nWidth = image->width;
	nHeight = image->height;
	if (image->nChannels == 3)
	{
		cvCvtColor(image, pGrayImage, CV_RGB2GRAY);
		cvSmooth(pGrayImage, pGrayImage, CV_MEDIAN, 3, 0, 0, 0);
	}
	else if (image->nChannels == 1)
	{
		cvSmooth(image, pGrayImage, CV_MEDIAN, 3, 0, 0, 0);
	}
	//timer.StartTime();
	//canny边缘算法
	cvCanny(pGrayImage, pEdgeImage, 120, 180);
	//sobel边缘算法
	hSobel(pGrayImage, pEdgeImage, pGradMat, pDirection, 10, 120);
	//亚像素分析
	hSubPixelEdge(pGradMat, pEdgeImage, pDirection, pSubEdgeMatH, pSubEdgeMatW);
	//timer.EndTime();
	//cout << "time:" << timer.GetTime() << "ms" << endl;
	cvShowImage("Image", pEdgeImage);

	////对生成的图像进行解析	
	int Rows = matImage.rows;//y
	int Cols = matImage.cols*matImage.channels();//x
	int x[100];
	int px = 0;
	int PixelDataof = 0;
	int sum = 0;
	double average = 0;
	for (int j = 0; j < Rows; j++) {
		uchar* data = matImage.ptr<uchar>(j);
		for (int i = 0; i < Cols; i++) {
			PixelDataof = data[i];
			if (PixelDataof > 130) {
				x[px] = i;
				px++;
				sum = sum + i;
				if (px > 100) {
					cout << "there are too many canny points" << endl;
				}
			}
		}
		//逐行计算平均点
		if (px) {
			average = sum / px;
		}
		point[j].cx = average;
		point[j].cy = j;
		average = 0;
		sum = 0;
		px = 0;
		memset(x, 0, px);
	}
	//误差标记函数double
	getErrorIdentifyDouble(matImage, point);

	//cvWaitKey(0);
	//cvReleaseImage(&image);
	//cvReleaseImage(&pGrayImage);
	//cvReleaseImage(&pEdgeImage);
	//cvReleaseMat(&pGradMat);
	//cvReleaseMat(&pSubEdgeMatH);
	//cvReleaseMat(&pSubEdgeMatW);
	//cvReleaseMat(&pDirection);
}



//基于高斯拟合的亚像素中心线检测算法  
void getGaussCenter(Mat matImage, MPoint *point, double maxError, double minError, int xRange) {
	Mat cloneImage = matImage.clone();
	//Mat OrgnImage = matImage.clone();
	//先运用canny检测得到初步中心线
	//int g_nCannyLowThreshold = 80;//canny检测低阈值
	//int minCanny = 200;//canny平均点筛选
	//Mat tmpImage, dstImage;
	//blur(cloneImage, tmpImage, Size(3, 3));
	//Canny(tmpImage, dstImage, g_nCannyLowThreshold, g_nCannyLowThreshold * 3);
	//namedWindow("canny function");
	//imshow("canny function", dstImage);
	int Rows = cloneImage.rows;
	int Cols = cloneImage.cols*cloneImage.channels();
	int *brightness;
	int threads = 2;//调用线程数
	brightness = new int[Rows];
	memset(brightness, 0, Rows);
	//getPeaker1(matImage, point);
#pragma omp parallel for num_threads(threads)
	for (int i = 0; i < Rows; i++)
	{
		uchar* data = matImage.ptr<uchar>(i);
		int MaxPixel = data[0];
		int MaxX(0);
		for (int j = 1; j < Cols; j++)
		{
			if (data[j] > MaxPixel) {
				MaxPixel = data[j];
				MaxX = j;
			}
		}
		point[i].y = i;
		point[i].x = MaxX;
		//point[i].bright = MaxPixel;
		brightness[i] = MaxPixel;
	}
	/*for (int i = 0; i < Rows; i++)
	{
		brightness[i] = point[i].bright;
	}*/
	//int x[100];
	//int px = 0; 


	//int sum = 0;
	//int average = 0;
	////getPeaker(matImage, brightness);
	//for (int j = 0; j < Rows; j++) {
	//	uchar* data = dstImage.ptr<uchar>(j);
	//	uchar* odata = matImage.ptr<uchar>(j);
	//	for (int i = 0; i < Cols; i++) {
	//		int PixelDataof = data[i];
	//		if (PixelDataof > minCanny) {//修改canny检测后的边缘阈值
	//			x[px] = i;
	//			px++;
	//			sum = sum + i;
	//			if (px > 100) {
	//				cout << "there are too many canny points" << endl;
	//			}
	//		}
	//	}
	//	//逐行计算平均点
	//	if (px) {
	//		average = sum * 1.0 / px;
	//	}
	//	point[j].x = average;
	//	point[j].y = j;
	//	brightness[j] = odata[average];
	//	
	//	average = 0;
	//	sum = 0;
	//	px = 0;
	//	memset(x, 0, px);
	//	//cout << "(" << point[j].cx << "," << point[j].cy << ")" << endl;
	//}

	//读取point中的值
//	int Cols = cloneImage.cols;//x
//	int Rows = cloneImage.rows;//y




	//逐行存储所有点的x坐标和亮度值以便分析 在此只存入高斯点
#pragma omp parallel for num_threads(threads)
	for (int i = 0; i < Rows; i++) {
		int PixelData;
		int Pixnum = 0;
		GPoint *gpoint;
		gpoint = new GPoint[Rows];
		Pixnum = 0;
		//高斯点选取 
		//watch.restart();
		uchar* data = matImage.ptr<uchar>(i);
		for (int j = 0; j < Cols; j++) {
			PixelData = data[j];
			//cout << PixelData << endl;
			//minerror和maxerror条件筛选高斯点  //后期在此处考虑xRange
			//cout << "condition1" << (PixelData > minError*brightness[i]) << endl;
			//cout << "condition2" << (PixelData < ((1 - maxError)*brightness[i]))<<endl;
			//cout << "condition3" << (abs(j - point[i].x) < xRange) << endl;

			if (PixelData > minError*brightness[i] && PixelData < ((1 - maxError)*brightness[i]) && (abs(j - point[i].x) < xRange)) {
				gpoint[Pixnum].x = j;
				gpoint[Pixnum].brightness = PixelData;
				Pixnum++;
			}
			if ((j - point[i].x) > xRange)
				break;
		}
		//watch.stop();
		//cout << "高斯点选取耗时:" << watch.elapsed() << endl;
		if (Pixnum >= 3) {
			int n = Pixnum;
			CvMat* X = cvCreateMat(n, 3, CV_64FC1);
			CvMat* Z = cvCreateMat(n, 1, CV_64FC1);
			//CvMat* XT = cvCreateMat(3, n, CV_64FC1);
			CvMat* B = cvCreateMat(3, 1, CV_64FC1);
			CvMat* SA = cvCreateMat(3, 3, CV_64FC1);
			CvMat* SAN = cvCreateMat(3, 3, CV_64FC1);
			CvMat* SC = cvCreateMat(3, n, CV_64FC1);
			//获取矩阵
			//watch.restart();
			getXZmatrix(X, Z, n, gpoint);
			//watch.stop();
			//cout << "矩阵获取耗时:" << watch.elapsed() << endl;
			//	/*for (int i = 0; i < n; i++) {
			//		for (int j = 0; j < 3; j++) {
			//			cout << cvmGet(X, i, j) << "\t";
			//		}
			//		cout << endl;
			//	}*/
			//	
			//	for (int i = 0; i < 3; i++) {
			//		for (int j = 0; j < n; j++) {
			//			cout << cvmGet(XT, i, j) << "\t";
			//		}
			//		cout << endl;
			//	}*/
			//乘法1
			//watch.restart();
			cvGEMM(X, X, 1, NULL, 0, SA, CV_GEMM_A_T);
			//watch.stop();
			//cout << "乘法1耗时:" << watch.elapsed() << endl;
			//for (int i = 0; i < 3; i++) {
			//		for (int j = 0; j < 3; j++) {
			//			cout << cvmGet(SA, i, j) << "\t";
			//		}
			//		cout << endl;
			//	}*/
			//矩阵求逆
			//watch.restart();
			cvInvert(SA, SAN, CV_LU);  //高斯消去法
			//watch.stop();
			//cout << "矩阵求逆耗时:" << watch.elapsed() << endl;
			/*for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {

					cout << cvmGet(SAN, i, j) << "\t";
				}
				cout << endl;
			}*/
			//矩阵乘法2
			//watch.restart();
			cvGEMM(SAN, X, 1, NULL, 0, SC, CV_GEMM_B_T);
			//watch.stop();
			//cout << "矩阵乘法2耗时:" << watch.elapsed() << endl;
			//	/*for (int i = 0; i < 3; i++) {
			//		for (int j = 0; j < n; j++) {
			//			cout << cvmGet(SC, i, j) << "\t";
			//		}
			//		cout << endl;
			//	}*/
			//	/*for (int i = 0; i < n; i++) {
			//		for (int j = 0; j < 1; j++) {
			//			cout << cvmGet(Z, i, j) << "\t";
			//		}
			//		cout << endl;
			//	}*/
			//矩阵乘法3
			//watch.restart();
			cvGEMM(SC, Z, 1, NULL, 0, B, 0);
			//watch.stop();
			//cout << "矩阵乘法3耗时:" << watch.elapsed() << endl;
			//	/*for (int i = 0; i < 3; i++) {
			//		cout << cvmGet(B, i, 0)<<"\t";
			//	}
			//	cout << endl;*/
			//结果计算
			//watch.restart();
			point[i].cx = (-cvmGet(B, 1, 0))*1.0 / (2 * cvmGet(B, 2, 0));
			point[i].bright = exp(cvmGet(B, 0, 0) - cvmGet(B, 1, 0)*cvmGet(B, 1, 0) / (4 * cvmGet(B, 2, 0)));
			//watch.stop();
			//cout << "结果计算耗时:" << watch.elapsed() << endl;
			cvReleaseMat(&X);
			cvReleaseMat(&Z);
			cvReleaseMat(&B);
			cvReleaseMat(&SA);
			cvReleaseMat(&SAN);
			cvReleaseMat(&SC);
		}
		else {
			point[i].cx = 0;
			point[i].bright = 0;
		}
		point[i].cy = i;
		delete[]gpoint;
	}

	//基于double的有阈值误差标记函数
	//getErrorIdentifyDoubleW(cloneImage, point, 0.15,0);


}

void getGaussCenter_hori(Mat matImage, MPoint *point, double maxError, double minError, int yRange)
{
	Mat cloneImage = matImage.clone();          //复制原图这里!
	int Rows = cloneImage.rows;
	int Cols = cloneImage.cols*cloneImage.channels();

	int *brightness;
	int threads = 8;		//调用线程数
	brightness = new int[Cols];
	memset(brightness, 0, Cols);
#pragma omp parallel for num_threads(threads)
	for (int j = 0; j < Cols; j++)
	{
		int MaxPixel = matImage.ptr<uchar>(0)[j];
		//cout << MaxPixel << endl;
		int MaxY(0);
		for (int i = 1; i < Rows; i++)
		{
			int tempPixel = matImage.ptr<uchar>(i)[j];
			//cout << tempPixel<< endl;
			if (tempPixel > MaxPixel)
			{
				MaxPixel = tempPixel;
				MaxY = i;
			}

		}
		point[j].x = j;
		point[j].y = MaxY;
		brightness[j] = MaxPixel;
		//cout << "(" << j << "," << MaxY << "):" << MaxPixel << endl;
	}
	/*完成：point中是每列最大像素位置信息，索引为列位置 ； brightness中是本列最大像素值，索引为列位置*/

#pragma omp parallel for num_threads(threads)
	for (int j = 0; j < Cols; j++)
	{
		int PixelData;
		int Pixnum = 0;
		GPoint *gpoint;
		gpoint = new GPoint[Cols];
		Pixnum = 0;
		for (int i = 0; i < Rows; i++)
		{
			PixelData = matImage.ptr<uchar>(i)[j];

			if (PixelData > minError*brightness[j] && PixelData < ((1 - maxError)*brightness[j]) && (abs(i - point[j].y) < yRange))
			{
				gpoint[Pixnum].x = i;
				gpoint[Pixnum].brightness = PixelData;
				Pixnum++;
			}
			if ((i - point[j].y) > yRange)
				break;
		}
		/*完成高斯点选取，gpoint索引为高斯点个数，存入y及像素*/

		if (Pixnum >= 3)
		{
			int n = Pixnum;
			CvMat *X = cvCreateMat(n, 3, CV_64FC1);
			CvMat *Z = cvCreateMat(n, 1, CV_64FC1);
			CvMat *B = cvCreateMat(3, 1, CV_64FC1);
			CvMat *SA = cvCreateMat(3, 3, CV_64FC1);
			CvMat *SAN = cvCreateMat(3, 3, CV_64FC1);
			CvMat *SC = cvCreateMat(3, n, CV_64FC1);

			getXZmatrix(X, Z, n, gpoint);
			cvGEMM(X, X, 1, NULL, 0, SA, CV_GEMM_A_T);
			cvInvert(SA, SAN, CV_LU);
			cvGEMM(SAN, X, 1, NULL, 0, SC, CV_GEMM_B_T);
			cvGEMM(SC, Z, 1, NULL, 0, B, 0);
			point[j].cy = (-cvmGet(B, 1, 0))*1.0 / (2 * cvmGet(B, 2, 0));
			point[j].bright = exp(cvmGet(B, 0, 0) - cvmGet(B, 1, 0)*cvmGet(B, 1, 0) / (4 * cvmGet(B, 2, 0)));

			//cout << "(" << j << "," << point[j].cy << ")" << endl;
			cvReleaseMat(&X);
			cvReleaseMat(&Z);
			cvReleaseMat(&B);
			cvReleaseMat(&SA);
			cvReleaseMat(&SAN);
			cvReleaseMat(&SC);
		}
		else {
			point[j].cy = 0;
			point[j].bright = 0;       //改canny
		}

		point[j].cx = j;
		delete[]gpoint;

	}
	//getErrorIdentifyDoubleW_hori(cloneImage, point, 0.15, 0);
}



void getErrorIdentifyDoubleW_hori(Mat matImage, MPoint *point, double doorin, int eHeight)
{
	int Rows = matImage.rows;
	int Cols = matImage.cols*matImage.channels();
	double error;
	for (int j = 1; j < Cols; j++)
	{
		if (abs(point[j].cy - point[j - 1].cy) > doorin)
		{
			line(matImage, Point(point[j].cx, point[j].cy - 30), Point(point[j].cx, point[j].cy + 30), Scalar(255, 100, 100), 2, 8, 0);
			line(matImage, Point(point[j].cx - 30, point[j].cy), Point(point[j].cx + 30, point[j].cy), Scalar(255, 100, 100), 2, 8, 0);
			error = point[j].cy - point[j - 1].cy;
			ostringstream oss;
			oss << error;
			string texterror = oss.str();
			putText(matImage, texterror, Point(point[j].cx, point[j].cy + 40), 2, 0.5, Scalar(255, 100, 100), 1, 8, 0);
		}
		namedWindow("error identification");
		imshow("error identification", matImage);
	}
}

void  getGaussCenter_horiColOnce(Mat matImage, MPoint *point, double maxError, double minError, int yRange, int Colonce) {
	Mat cloneImage = matImage.clone();          //复制原图这里!
	int Rows = cloneImage.rows;
	int Cols = cloneImage.cols*cloneImage.channels();

	//int *brightness;
	int threads = 8;		//调用线程数
	//brightness = new int[Cols];
	//memset(brightness, 0, Cols);

	//每Colonce列数据转置后存入二维数组
#pragma omp parallel for num_threads(threads)
	for (int k = 0; k < Cols / Colonce; k++) {
		int** array_ = new int*[Colonce];
		for (int i = 0; i < Colonce; i++)
			array_[i] = new int[Rows];
		for (int j = 0; j < Rows; j++) {
			uchar* data = matImage.ptr<uchar>(j);
			//逐行分Colonce列存入
			for (int i = 0; i < Colonce; i++) {
				array_[i][j] = data[k*Colonce + i];
				//cout << "(" << k*Colonce + i << "," <<j << "):" << array_[i][j] << endl;
			}
		}
		//取每列最大值及位置
		for (int i = 0; i < Colonce; i++) {
			int MaxPixel = array_[i][0];
			int MaxY = 0;
			for (int j = 1; j < Rows; j++) {
				if (array_[i][j] > MaxPixel) {
					MaxPixel = array_[i][j];
					MaxY = j;
				}
			}
			point[k*Colonce + i].x = k * Colonce + i;
			point[k*Colonce + i].y = MaxY;
			point[k*Colonce + i].bright = MaxPixel;
			//cout<< "(" << point[k*Colonce + i].x << "," << point[k*Colonce + i].y << "):"<< point[k*Colonce + i].bright << endl;
		}
		//高斯点筛选
		for (int i = 0; i < Colonce; i++) {
			int Pixnum = 0;
			GPoint *gpoint;
			gpoint = new GPoint[Rows];
			for (int j = 0; j < Rows; j++) {
				if ((array_[i][j] > minError*point[k*Colonce + i].bright)
					&& (array_[i][j] < (1 - maxError)*point[k*Colonce + i].bright)
					&& (abs(j - point[k*Colonce + i].y) < yRange))
				{
					gpoint[Pixnum].x = k * Colonce + i;
					gpoint[Pixnum].brightness = array_[i][j];
					Pixnum++;
					//cout << "(" << gpoint[Pixnum].x << "," << gpoint[Pixnum].brightness << ")" << endl;
				}
				if ((j - point[k*Colonce + i].y) > yRange)
					break;
			}


			//矩阵运算
			if (Pixnum >= 3) {
				int n = Pixnum;
				CvMat *X = cvCreateMat(n, 3, CV_64FC1);
				CvMat *Z = cvCreateMat(n, 1, CV_64FC1);
				CvMat *B = cvCreateMat(3, 1, CV_64FC1);
				CvMat *SA = cvCreateMat(3, 3, CV_64FC1);
				CvMat *SAN = cvCreateMat(3, 3, CV_64FC1);
				CvMat *SC = cvCreateMat(3, n, CV_64FC1);
				getXZmatrix(X, Z, n, gpoint);
				cvGEMM(X, X, 1, NULL, 0, SA, CV_GEMM_A_T);
				cvInvert(SA, SAN, CV_LU);
				cvGEMM(SAN, X, 1, NULL, 0, SC, CV_GEMM_B_T);
				cvGEMM(SC, Z, 1, NULL, 0, B, 0);
				point[k*Colonce + i].cy = (-cvmGet(B, 1, 0))*1.0 / (2 * cvmGet(B, 2, 0));
				point[k*Colonce + i].bright = exp(cvmGet(B, 0, 0) - cvmGet(B, 1, 0)*cvmGet(B, 1, 0) / (4 * cvmGet(B, 2, 0)));
				cvReleaseMat(&X);
				cvReleaseMat(&Z);
				cvReleaseMat(&B);
				cvReleaseMat(&SA);
				cvReleaseMat(&SAN);
				cvReleaseMat(&SC);
			}
			//否则执行简单算法
			else {
				point[k*Colonce + i].cy = 0;
				point[k*Colonce + i].bright = 0;
			}
			point[k*Colonce + i].cx = k * Colonce + i;
			delete[]gpoint;

		}
		for (int i = 0; i < Colonce; i++) {
			delete[] array_[i];
		}

		delete[] array_;

	}

	//getErrorIdentifyDoubleW_hori(cloneImage, point,0.15,0);

}

//f(n,x)用来返回x的n次方
double f(double x, int n) {
	double y = 1.0;
	if (n == 0) return 1.0;
	else {
		for (int i = 0; i < n; i++) {
			y *= x;
		}
		return y;
	}
}

double fx(int x, int n) {
	int y;
	if (n == 0)
	{
		y = 1;
	}
	else if (n == 1) {
		y = x;
	}
	else if (n == 2) {
		y = x * x;
	}
	return y;
}

//X,Z矩阵的生成 //X矩阵 Z矩阵  数据点个数  输入GPoint
int getXZmatrix(CvMat* X, CvMat* Z, int n, GPoint *gpoint) {
	//n个数据点 以n行形式存入
	for (int i = 0; i < n; i++) {
		//顺序存入 1  x  x^2
		//double* xData = (double*)(X->data.ptr + i * X->step);
		//double* zData = (double*)(X->data.ptr + i * X->step);
		for (int j = 0; j < 3; j++) {
			cvmSet(X, i, j, fx(gpoint[i].x, j));
			//xData[j] = fx(gpoint[i].x, j);
			//cout << j << endl;
		}
		//y存入Z矩阵
		cvmSet(Z, i, 0, log(gpoint[i].brightness));
		//zData[0] = log(gpoint[i].brightness);
	}
	return 1;
}


//基于double的有阈值误差标记函数
void getErrorIdentifyDoubleW(Mat matImage, MPoint *point, double doorin, int eHeight) {
	int Rows = matImage.rows;//y
	//int Cols = matImage.cols;
	int Cols = matImage.cols*matImage.channels();//x
	//int div = 64;
	double error;
	for (int j = 0; j < Rows; j++) {
		//point[j].errorup = point[j].cx - point[j - 1].cx;
		if (abs(point[j].cx - point[j - 1].cx) > doorin) {
			line(matImage, Point((point[j].cx - 30), point[j].cy), Point((point[j].cx + 30), point[j].cy), Scalar(255, 100, 100), 2, 8, 0);
			line(matImage, Point(point[j].cx, point[j].cy - 30), Point(point[j].cx, point[j].cy + 30), Scalar(255, 100, 100), 2, 8, 0);
			error = point[j].cx - point[j - 1].cx;
			ostringstream oss;
			oss << error;
			string texterror = oss.str();
			putText(matImage, texterror, Point(point[j].cx + 40, point[j].cy), 2, 0.5, Scalar(255, 100, 100), 1, 8, 0);
		}
	}
	namedWindow("error identification");
	imshow("error identification", matImage);
}

//基于double的误差标记函数
void getErrorIdentifyDouble(Mat matImage, MPoint *point) {
	int Rows = matImage.rows;//y
	//int Cols = matImage.cols;
	int Cols = matImage.cols*matImage.channels();//x
	//int div = 64;
	double error;
	for (int j = 0; j < Rows; j++) {
		if (point[j].cx != point[j - 1].cx) {
			line(matImage, Point((point[j].cx - 30), point[j].cy), Point((point[j].cx + 30), point[j].cy), Scalar(255, 100, 100), 2, 8, 0);
			line(matImage, Point(point[j].cx, point[j].cy - 30), Point(point[j].cx, point[j].cy + 30), Scalar(255, 100, 100), 2, 8, 0);
			error = point[j].cx - point[j - 1].cx;
			ostringstream oss;
			oss << error;
			string texterror = oss.str();
			putText(matImage, texterror, Point(point[j].cx + 40, point[j].cy), 2, 0.5, Scalar(255, 100, 100), 1, 8, 0);
		}
	}
	namedWindow("error identification");
	imshow("error identification", matImage);
}

//双精度平均值计算
double average(int *x, int len) {
	int sum = 0;
	int num = 0;
	double average;
	for (int i = 0; i < len; i++) {
		if (x[i]) {
			sum = sum + x[i];
			num++;
		}
	}
	average = sum / num;
	sum = 0;
	num = 0;
	return average;

}



//误差标记函数
void getErrorIdentifyInt(Mat matImage, MPoint *point) {
	int Rows = matImage.rows;//y
	//int Cols = matImage.cols;
	int Cols = matImage.cols*matImage.channels();//x
	//int div = 64;
	int error;
	for (int j = 0; j < Rows; j++) {
		if (point[j].x != point[j - 1].x) {
			line(matImage, Point((point[j].x - 30), point[j].y), Point((point[j].x + 30), point[j].y), Scalar(255, 100, 100), 2, 8, 0);
			line(matImage, Point(point[j].x, point[j].y - 30), Point(point[j].x, point[j].y + 30), Scalar(255, 100, 100), 2, 8, 0);
			error = point[j].x - point[j - 1].x;
			std::ostringstream oss;
			oss << error;
			std::string texterror = oss.str();
			putText(matImage, texterror, Point(point[j].x + 40, point[j].y), 2, 0.5, Scalar(255, 100, 100), 1, 8, 0);
		}
	}
	namedWindow("error identification");
	imshow("error identification", matImage);
}

//被giveup的寻址method1
void getDiff1test(Mat matImage, MPoint *point) {
	int Rows = matImage.rows;//y
	//int Cols = matImage.cols;
	int Cols = matImage.cols*matImage.channels();//x
	//int div = 64;
	int MaxPixel;
	int error = 0;
	int errormax = 30;
	int errormin = 2;
	MPoint samepoint, startpoint;
	samepoint.x = 0;
	samepoint.y = 0;
	samepoint.bright = 0;
	samepoint.Pixnum = 0;
	for (int j = 1; j < Rows; j++) {

		if (((point[j].x) == (point[j - 1].x))) {
			;
		}
		else if ((point[j].x) != (point[j - 1].x) && !error) {   //第一次遇到转点
			error++;
			/*samepoint.x = point[j].x;
			samepoint.y = point[j].y;
			samepoint.bright = point[j].bright;
			samepoint.Pixnum = point[j].Pixnum;*/
			samepoint = point[j];
		}
		else if ((point[j].x != samepoint.x) && error) {
			error++;
		}
		else if (((point[j].x) == samepoint.x) && error) {
			//判断error是否符合标准
			if (error >= errormin && error <= errormax) {
				//cvLine画线标定
				line(matImage, Point((point[j].x - 30), point[j].y), Point((point[j].x + 30), point[j].y), Scalar(255, 100, 100), 2, 8, 0);
				line(matImage, Point(point[j].x, point[j].y - 30), Point(point[j].x, point[j].y + 30), Scalar(255, 100, 100), 2, 8, 0);
				line(matImage, Point((samepoint.x - 30), samepoint.y), Point((samepoint.x + 30), samepoint.y), Scalar(255, 100, 100), 2, 8, 0);
				line(matImage, Point(samepoint.x, samepoint.y - 30), Point(samepoint.x, samepoint.y + 30), Scalar(255, 100, 100), 2, 8, 0);
			}
			error = 0;
			samepoint.x = 0;
			samepoint.y = 0;
			samepoint.bright = 0;
			samepoint.Pixnum = 0;
		}
	}


	namedWindow("error identification");
	imshow("error identification", matImage);
}
