#pragma once
//ImProcess.h
//2009-5-29……
#ifndef IMPROCESS_H
#define IMPROCESS_H

#define	PI			3.14159		
#define COUTINFO    1

#include <iostream>
#include "cxcore.h"		       //OpenCV头文件
#include "highgui.h"
#include "cv.h" 

//#pragma comment (lib,"highgui.lib")
//#pragma comment (lib,"cxcore.lib")
////#pragma comment (lib,"cv.lib")

class CHTimer
{
public:
	CHTimer();
	~CHTimer();
	void StartTime();
	void EndTime();
	double GetTime();
protected:

private:
	long  m_lStartTime;
	long  m_lEndTime;
	long  m_lPersecond;
	double	  m_fTime;
};

int inline AbsInt(int a)
{
	if (a > 0)
		return a;
	else
		return -a;
}
double inline square(double a)
{
	return a * a;
}
//滤波算法
void hFilter(IplImage* src,   //源图像
	IplImage* dst,   //目标图像，即滤波后的图像
	int nThreshold); //阈值，调用时nThreshold=0，函数使用默认值，否则以该值作为阈值
//边缘检测算法,sobel算子
void hSobel(IplImage* src,    //原始图像
	IplImage* dst,   //边缘图像
	CvMat* GradImage, //梯度图
	CvMat* Direction, //垂直于边缘梯度的方向矩阵
	int nThreshold1,  //二值化阈值1 nThreshold1<=nThreshold2
	int nThreshold2);//二值化阈值2 nThreshold1<=nThreshold2
void hSubPixelEdge(CvMat* GradImage,   //灰度图像
	IplImage* EdgeImage,//边缘图像
	CvMat* direction,   //梯度方向矩阵
	CvMat* SubEdgeMatH, //亚像素边缘H值
	CvMat* SubEdgeMatW);//亚像素边缘H值
#endif