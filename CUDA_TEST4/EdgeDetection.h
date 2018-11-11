#pragma once
#include "pch.h"
#include <iostream>
#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "cv.h"
#include <process.h>
#include "CameraApi.h"
#include "LaserRange.h"
#include "windows.h"
#include "math.h"
#include "cstdlib"
#include "sstream"
#include "ImProcess.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"


using namespace std;
using namespace cv;



//边缘测试算法1
void getPeaker1(Mat matImage, MPoint *point);
//Canny opencv 边缘检测
void getcanny(Mat matImage, MPoint *point);
//laplician 边缘检测
void getlaplacian(Mat matImage, MPoint *point);
//Sobel 边缘检测
void getsobel(Mat matImage, MPoint *point);
//亚像素分析函数（待完善）
void getdoublepixel(Mat matImage, MPoint *point);
//基于高斯拟合的亚像素中心线检测算法  
void getGaussCenter(Mat matImage, MPoint *point, double maxError, double minError, int xRange);
void getGaussCenter_hori(Mat matImage, MPoint *point, double maxError, double minError, int yRange);//横线
void  getGaussCenter_horiColOnce(Mat matImage, MPoint *point, double maxError, double minError, int yRange, int Colonce);//横线ColOnce
//X,Z矩阵的生成 //X矩阵 Z矩阵  数据点个数  输入GPoint
int getXZmatrix(CvMat* X, CvMat* Z, int n, GPoint *gpoint);
//f(n,x)用来返回x的n次方
double f(double x, int n);
//高斯主元法求解方程组
int GaussMajorEquation(double **a, int n, double *b, double *p, double dt);
//动态生成数组
double** create(int a, int b);
//最小二乘拟合//    m A矩阵的大小
void MinDoubleFit(double x[], double y[], int n, double a[], int m);
//计算最小二乘解的平方误差
double ErrorSqrt(double x[], double y[], int n, double a[], int m);
//基于double的带阈值误差标记函数
void getErrorIdentifyDoubleW(Mat matImage, MPoint *point, double doorin, int eHeight);
void getErrorIdentifyDoubleW_hori(Mat matImage, MPoint *point, double doorin, int eHeight);//横线
//基于double的误差标记函数
void getErrorIdentifyDouble(Mat matImage, MPoint *point);
//双精度平均值计算
double average(int *x, int len);
//误差标记函数
void getErrorIdentifyInt(Mat matImage, MPoint *point);
//被giveup的寻址method1
void getDiff1test(Mat matImage, MPoint *point);