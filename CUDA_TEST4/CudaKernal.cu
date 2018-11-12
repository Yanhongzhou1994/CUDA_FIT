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


__global__ void GetGaussPointCuda(PtrStepSz<int> src,MPoint *point, double maxError, double minError, int yRange, int Colonce,int Rows,int Cols) {
	int threadId = threadIdx.x;
	//printf("%d\n",threadId);
	int *gpu_cr = new int [Rows*Colonce];
	//逐行存入数组
	for (int j = 0; j < Rows; j++)
	{
		for (int i = 0; i < Colonce; i++)
		{
			gpu_cr[i*Rows+j] = src(threadId*Colonce + i,j);
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
		//GPoint *gpoint;
		point[i].gpoint = new GPoint[Rows];
		//point[i].gpoint = new GPoint[Rows];
		for (int j = 0; j < Rows; j++)
		{
			if ((gpu_cr[Rows*i + j] > minError*point[threadId*Colonce + i].bright)
				&& (gpu_cr[Rows*i + j] < (1 - maxError)*point[threadId*Colonce + i].bright)
				&& (abs(j - point[threadId*Colonce + i].y) < yRange))
			{
				point[i].gpoint[Pixnum].x = threadId * Colonce + i;
				point[i].gpoint[Pixnum].brightness = gpu_cr[Rows*i + j];
				Pixnum++;
			}
			if ((j - point[threadId*Colonce + i].y) < yRange)
				break;
		}

		/*
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
			//求X转置
			__shared__ int *XT;
			XT = new int[Pixnum* 3];
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < Pixnum; j++)
				{
					XT[i*Pixnum + j] = X[j * 3 + i];
				}
			}
			//求XT*X结果
			__shared__ int *SA;
			SA = new int[3 * 3];
			for (int m = 0; i < 3; i++)
			{
				for (int s = 0; s < 3; s++)
				{
					for (int n = 0; n < Pixnum; n++)
					{
						SA[m * 3 + s] = XT[m*Pixnum + n] * X[n * 3 + s];
					}
				}
			}
			//求SA逆矩阵
			__shared__ int *SAN;
			SAN = new int[3 * 3];

			
		}*/
	}
	delete &gpu_cr;	
	__syncthreads();
}

//#define N 3
//__global__ void MatAdd(const int **A, const int **B, int **C)
//{
//	int i = threadIdx.x;
//	int j = threadIdx.y;
//	C[i][j] = A[i][j] + B[i][j];
//	//__syncthreads();
//}
__global__ void getEveryPixel(PtrStep<uchar> gpuMat, PtrStep<uchar> outMat, int Rows, int Cols) {
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.y*blockIdx.y;
	if (i < Cols&&j < Rows)
	{
		//outmat = 
	}
}


//extern "C" void GetGaussFitCuda(GpuMat gpuMat, MPoint *point, double maxError, double minError, int yRange, int Colonce);

extern "C"
void CudaGuassHC(Mat matImage, MPoint *point, double maxError, double minError, int yRange, int Colonce) {

	int Rows = matImage.rows;
	int Cols = matImage.cols;// matImage.channels();
	InputArray inputMat(matImage);
	GpuMat gpuMat(Rows, Cols, CV_8UC1);
	gpuMat.upload(matImage);
	//for (int j = 0; j < Rows; j++) {
	//	//uchar* data = gpuMat.ptr<uchar>(j);
	//	for (int i = 0; i < Cols; i++) {
	//		int datt = gpuMat.ptr<uchar>(j)[i];
	//		//cout << "(" << i << "," <<j << "):" << datt << endl;
	//		printf("(%d,%d):%d\n", i, j, datt);
	//	}
	//}

	//dim3 blocks_all(1);
	dim3 threads_all(Cols / Colonce);

	GetGaussPointCuda << <1, threads_all >> > (gpuMat, point, maxError, minError, yRange, Colonce, Rows, Cols);
	for (int i = 0; i < Cols; i++)
	{
		//cout << "("<<point[i].x<<","<< point[i].y<<"):"<< point[i].bright << endl;
		printf("(%d,%d):%d\n", point[i].x, point[i].y, point[i].bright);
	}


	gpuMat.release();


	////test for krenal global
	//int numBlocks = 1;
	//dim3 threadsPerBlock(3, 3);
	//const int A[3][3] = { {3,3,3},{2,2,2},{1,1,1} };
	//const int B[3][3] = { {3,3,3},{2,2,2},{1,1,1} };
	//int C[3][3] = { {0,0,0},{0,0,0},{0,0,0} };
	//const int **a = &A[0][0];
	//const int *b = &B[0][0];
	//int *c = &C[0][0];
	//MatAdd << <numBlocks, threadsPerBlock >> > (a, b, c);
	//for (int i = 0; i < 3; i++)
	//{
	//	for (int j = 0; j < 3; j++) {
	//		cout << C[i][j] << "   ";
	//	}
	//	cout << endl;
	//}
	//getchar();


}


