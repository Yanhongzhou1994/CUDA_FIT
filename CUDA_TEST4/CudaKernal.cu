#include "iostream"
#include "cuda_runtime.h"
#include "cuda.h"
#include "cublas.h"
#include "cublas_api.h"
#include "pch.h"
//#include "stdafx.h"
#include "cv.h"
//#include <process.h>
//#include "CameraApi.h"
#include "LaserRange.h"
//#include "afxwin.h"
//#include "windows.h"
#include "math.h"
//#include "cstdlib"
//#include "sstream"
//#include "ImProcess.h"
//#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
#include "CudaTest.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
//#include <stdio.h>
//#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
//#include "CudaKernal.cuh"

using namespace std;
using namespace cv;
using namespace cv::cuda;

//extern "C" void GetGaussFitCuda(GpuMat gpuMat, MPoint *point, double maxError, double minError, int yRange, int Colonce);

//矩阵乘法 a[M][N]*b[N][S]
__device__ void MatrixMul(int *a, int *b, int *result, int M, int N, int S) {
	int row = threadIdx.x;
	int col = threadIdx.y;

}
//矩阵存入
__device__ void SetElementX( GPoint *gpoint, int *X, int Pixnum) {
	int threadId = threadIdx.x;

}
__device__ void SetElementZ(int *Z, int) {

}
//矩阵求逆 m阶a矩阵
__device__ void MatrixInvert(int *a, int *at, int M) {

}
//矩阵转置

__global__ void GetGaussFitCuda(PtrStepSz<uchar> src,MPoint *point, double maxError, double minError, int yRange, int Colonce,int Rows,int Cols) {
	int threadId = threadIdx.x;
	int *gpu_cr = new int [Rows*Colonce];
	//逐行存入数组
	for (int j = 0; j < Rows; j++)
	{
		for (int i = 0; i < Colonce; i++)
		{
			gpu_cr[i*Rows+j] = src(threadId*i, j);
		}
	}
	
	//取每列最大值位置
	for (int i = 0; i < Colonce; i++) {
		int MaxPixel = gpu_cr[Rows*i];
		int MaxY = 0;
		for (int j = 1; j < Rows; j++)
		{
			if (gpu_cr[Rows*i + j] > MaxPixel)
			{
				MaxPixel = gpu_cr[Rows*i + j];
				MaxY = j;
			}
		}
		point[threadId*Colonce + i].x = threadId * Colonce + i;
		point[threadId*Colonce + i].y = MaxY;
		point[threadId*Colonce + i].bright = MaxPixel;
	}

	//高斯点筛选
	for (int i = 0; i < Colonce; i++)
	{
		int Pixnum = 0;
		GPoint *gpoint;
		gpoint = new GPoint[Rows];
		for (int j = 0; j < Rows; j++)
		{
			if ((gpu_cr[Rows*i + j] > minError*point[threadId*Colonce + i].bright)
				&& (gpu_cr[Rows*i + j] < (1 - maxError)*point[threadId*Colonce + i].bright)
				&& (abs(j - point[threadId*Colonce + i].y) < yRange))
			{
				gpoint[Pixnum].x = threadId * Colonce + i;
				gpoint[Pixnum].brightness = gpu_cr[Rows*i + j];
				Pixnum++;
			}
			if ((j - point[threadId*Colonce + i].y) < yRange)
				break;
		}


		//矩阵运算
		if (Pixnum >= 3)
		{
			__shared__ int *X;
			X = new int[Pixnum * 3];
			__shared__ int *Z;
			Z = new int[Pixnum];
			//计算能力<3.5 不能嵌套并行核函数
			//dim3 blockSEX(1, 0, 0);
			//dim3 threadSEX(Pixnum, 0, 0);
			//存入X、Z矩阵
			//SetElementX << <blockSEX, threadSEX >> > (gpoint, X, Pixnum);
			//存入X矩阵(n*3) Z矩阵（n*1)
			for (int i = 0; i < Pixnum; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					if (j = 0)
					{
						X[i * 3 + j] = 1;
					}
					if (j = 1)
					{
						X[i * 3 + j] = gpoint[i].x;
					}
					if (j = 2)
					{
						X[i * 3 + j] = gpoint[i].x*gpoint[i].x;
					}
				}
				Z[i] = gpoint[i].brightness;
			}
			__shared__ int *XT;
			XT = new int[Pixnum * 3];
			for (int i = 0; i < Pixnum; i++)
			{

			}
			
		}
	}
	delete gpu_cr;	
}



//extern "C" void GetGaussFitCuda(GpuMat gpuMat, MPoint *point, double maxError, double minError, int yRange, int Colonce);
extern "C"
void CudaGuassHC(Mat matImage, MPoint *point, double maxError, double minError, int yRange, int Colonce) {
	int Rows = matImage.rows;
	int Cols = matImage.cols*matImage.channels();
	GpuMat gpuMat;
	gpuMat.upload(matImage);
	dim3 blocks_all(1,0,0);
	dim3 threads_all(Cols / Colonce,0,0);
	GetGaussFitCuda<<<1,threads_all>>>(gpuMat, point, maxError, minError, yRange, Colonce,Rows,Cols);
	
	
}


