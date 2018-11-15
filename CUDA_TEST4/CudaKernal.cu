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
cudaError_t checkCudaError(cudaError_t CudaFunction,const char* ident) {
	cudaError_t err = CudaFunction;
	if (err != cudaSuccess) {
		fprintf(stderr, "%s \t cudaError:%s\n",ident,cudaGetErrorString(cudaGetLastError()));
	}
	return err;
}

//Coloncel行扫描得点存储
__global__ void GetGaussPointCuda(PtrStepSz<uchar1> src, MPoint *point, int **gpu_data, int Colonce, int Rows, int Cols) {
	int threadId = threadIdx.x;
	//printf("%d\n",threadId);
	//__shared__ int *gpu_cr;
	//gpu_cr = new int [Rows*Cols];
	//逐行存入数组
	for (int j = 0; j < Rows; j++)
	{
		for (int i = 0; i < Colonce; i++)
		{
			gpu_data[i + threadId * Colonce][j] = (int)src(j, threadId*Colonce + i).x;
		}
	}
	//int i = 0, j = 0;
	//gpu_cr[i*Rows + j] = *((int*)&src( j, threadId*Colonce + i));

	//取每列最大值位置
	for (int i = 0; i < Colonce; i++) {
		int MaxPixel = gpu_data[i + threadId * Colonce][0];
		//printf("the first pixel is %d \n", MaxPixel);
		int MaxY = 0;
		for (int j = 1; j < Rows; j++)
		{
			if (gpu_data[i + threadId * Colonce][j] > MaxPixel)
			{
				MaxPixel = gpu_data[i + threadId * Colonce][j];
				MaxY = j;
			}
		}

		point[threadId*Colonce + i].x = threadId * Colonce + i;
		point[threadId*Colonce + i].y = MaxY;
		point[threadId*Colonce + i].bright = MaxPixel;
	}
	__syncthreads();
}

//按列筛选并处理高斯点
__global__ void GetGaussFitRes(MPoint *point, int **gpu_data, double maxError, double minError, int yRange, int Rows, int Cols) 
{
	//通过块并行解决一个block内thread不够用的问题
	int threadId = blockIdx.x*blockDim.x + threadIdx.x;
	//判断以确定该线程有可处理数据
	if (threadId < Cols)
	{
		////高斯点存储申请
		//int *y; //存储高斯点在每列的行位置
		//int *br; //存储高斯点的值
		GPoint *gpoint = new GPoint[2 * yRange];
		int Pixnum = 0; //统计高斯点个数
		//确定上下界位置 减少计算次数
		double minLine = minError * point[threadId].bright;
		double maxLine = (1-maxError) * point[threadId].bright;
		//高斯点筛选
		for (int i = (point[threadId].y - yRange); i < (point[threadId].y+yRange+1); i++)
		{
			if ((gpu_data[threadId][i] > minLine)&&(gpu_data[threadId][i] < maxLine))
			{
				gpoint[Pixnum].x = i;
				gpoint[Pixnum].brightness = gpu_data[threadId][i];
				Pixnum++;
			}
		}
		point[threadId].Pixnum = Pixnum;
		/*
		if (Pixnum > 3) {
			//运算矩阵申请
			int n = Pixnum;
			//X矩阵（1 x x^2）  n*3
			double **X = new double*[n];
			for (int i = 0; i < n; i++) {
				X[i] = new double[3];
			}
			//XT矩阵 X的转置  3*n
			double **XT = new double*[3];
			for (int i = 0; i < 3; i++) {
				XT[i] = new double[n];
			}
			//Z矩阵(brightness) n*1
			double *Z = new double[n];
			//B矩阵（结果） 3*1
			double *B = new double[3];
			//SA矩阵 （XT*X） 3*3
			double **SA = new double*[3];
			for (int i = 0; i < 3; i++) {
				SA[i] = new double[3];
			}
			//SAN矩阵 SA的逆矩阵  3*3
			double **SAN = new double*[3];
			for (int i = 0; i < 3; i++) {
				SAN[i] = new double[3];
			}
			//SC矩阵  SAN*XT 3*n
			double **SC = new double *[3];
			for (int i = 0; i < 3; i++) {
				SC[i] = new double[n];
			}
			//存入X矩阵和Z矩阵 顺手存入转置XT
			for (int i = 0; i < n; i++) {
				X[i][0] = 1;
				X[i][1] = gpoint[i].x;
				X[i][2] = gpoint[i].x*gpoint[i].x;
				Z[i] = gpoint[i].brightness;
				XT[0][i] = 1;
				XT[1][i] = gpoint[i].x;
				XT[2][i] = gpoint[i].x*gpoint[i].x;
			}
			//计算XT*X
			for (int m = 0; m < 3; m++) {
				for (int s = 0; s < 3; s++) {
					SA[m][s] = 0;
					for (int i = 0; i < n; i++) {
						SA[m][s] += XT[m][i] * X[i][s];
					}
				}
			}
			//计算SAN
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					SAN[i][j] = 0;
				}
			}
			//计算SC = SAN*XT
			for (int m = 0; m < 3; m++) {
				for (int s = 0; s < n; s++) {
					SC[m][s] = 0;
					for (int i = 0; i < 3; i++) {
						SC[m][s] += SAN[m][i] * XT[i][s];
					}
				}
			}
			//计算B = SC*Z
			for (int m = 0; m < 3; m++) {
				B[m] = 0;
				for (int i = 0; i < n; i++) {
					B[m] += SC[m][i] * Z[i];
				}
			}
			//解析B
			point[threadId].cx = threadId;
			point[threadId].cy = (-B[1]) / (2 * B[2]);
			point[threadId].bright = exp(B[0] - B[1] * B[1] / (4 * B[2]));

		}
		else
		{
			point[threadId].cx = threadId;
			point[threadId].cy = 0;
			point[threadId].bright = 0;
		}
		
		*/
		
		delete[] gpoint;

	}
	else
		return;

}
	
	//高斯点筛选
	//for (int i = 0; i < Colonce; i++)
	//{
	//	int Pixnum = 0;
	//	//GPoint *gpoint;
	//	//point[threadId*Colonce+i].gpoint = new GPoint[Rows];
	//	//point[i].gpoint = new GPoint[Rows];
	//	for (int j = 0; j < Rows; j++)
	//	{
	//		if ((gpu_cr[Rows*i + j] > minError*point[threadId*Colonce + i].bright)
	//			&& (gpu_cr[Rows*i + j] < (1 - maxError)*point[threadId*Colonce + i].bright)
	//			&& (abs(j - point[threadId*Colonce + i].y) < yRange))
	//		{
	//			point[threadId*Colonce + i].gpoint[Pixnum].x = threadId * Colonce + i;
	//			point[threadId*Colonce + i].gpoint[Pixnum].brightness = gpu_cr[Rows*i + j];
	//			Pixnum++;
	//		}
	//		if ((j - point[threadId*Colonce + i].y) < yRange)
	//			break;
	//	}
	//	point[threadId*Colonce + i].Pixnum = Pixnum;

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
	//}
	//delete &gpu_cr;
	

//#define N 3
//__global__ void MatAdd(const int **A, const int **B, int **C)
//{
//	int i = threadIdx.x;
//	int j = threadIdx.y;
//	C[i][j] = A[i][j] + B[i][j];
//	//__syncthreads();
//}



//extern "C" void GetGaussFitCuda(GpuMat gpuMat, MPoint *point, double maxError, double minError, int yRange, int Colonce);

extern "C"
void CudaGuassHC(Mat matImage, MPoint *point, double maxError, double minError, int yRange, int Colonce) {

	int Rows = matImage.rows;
	int Cols = matImage.cols;// *matImage.channels();
	//InputArray inputMat(matImage);
	//for (int j = 0; j < Rows; j++) {
	//	//uchar* data = gpuMat.ptr<uchar>(j);
	//	for (int i = 0; i < Cols; i++) {
	//		int datt = inputMat.ptr<uchar>(j)[i];
	//		//cout << "(" << i << "," <<j << "):" << datt << endl;
	//		printf("(%d,%d):%d\n", i, j, datt);
	//	}
	//}
	//cout << Cols << endl;
	GpuMat gpuMat(matImage);
	//gpuMat.upload(matImage);
	//for (int j = 0; j < Rows; j++) {
	//	//uchar* data = gpuMat.ptr<uchar>(j);
	//	for (int i = 0; i < Cols; i++) {
	//		int datt = gpuMat.ptr<uchar>(j)[i];
	//		//cout << "(" << i << "," <<j << "):" << datt << endl;
	//		printf("(%d,%d):%d\n", i, j, datt);
	//	}
	//}
	//结构体指针上传
	MPoint *gpu_point;
	//gpu_point = new MPoint[Cols];	
	checkCudaError(cudaMalloc((void**)&gpu_point, sizeof(MPoint)*Cols), "malloc error1");
	//显存图像缓存矩阵
	int **gpu_data;
	int *gpu_data_d;
	int **cpu_data = (int**)malloc(sizeof(int*)*Cols);
	int *cpu_data_d = (int*)malloc(sizeof(int)*Cols*Rows);
	checkCudaError(cudaMalloc((void**)&gpu_data, Cols * sizeof(int**)), "malloc error2");
	checkCudaError(cudaMalloc((void**)&gpu_data_d, Cols *Rows * sizeof(int)), " malloc error2");
	for (int i = 0; i < Cols; i++) {
		cpu_data[i] = gpu_data_d + Rows * i;
		//首地址赋值 将一维矩阵转为二维
	}
	checkCudaError(cudaMemcpy(gpu_data, cpu_data, sizeof(int*)*Cols, cudaMemcpyHostToDevice), "memcpy error1");
	checkCudaError(cudaMemcpy(gpu_data_d, cpu_data_d, sizeof(int)*Rows*Cols, cudaMemcpyHostToDevice), "memcpy error1");  

	/*if (cudaSuccess != cudaMemcpy(gpu_point, point, sizeof(point)*Cols, cudaMemcpyHostToDevice)) {
		printf("cuda memcpy up error1!\n");
	}*/
	
	//dim3 threads_all(Cols / Colonce);
	//每colonce列统一存入 
	GetGaussPointCuda << <1, Cols/Colonce >> > (gpuMat, gpu_point, gpu_data, Colonce, Rows, Cols);
	cudaDeviceSynchronize();
	//规划并行流  之后设计为只规划一次
	int Blocknum, Threadnum;
	if (Cols > 1024) {
		Blocknum = Cols / 1024 + 1;
		Threadnum = 1024;
	}
	else {
		Blocknum = 1;
		Threadnum = Cols;
	}
	//进行高斯拟合
	GetGaussFitRes << <Blocknum, Threadnum >> > (gpu_point, gpu_data, maxError, minError, yRange, Rows, Cols);
	checkCudaError(cudaMemcpy(point, gpu_point, sizeof(MPoint)*Cols, cudaMemcpyDeviceToHost), "memcpy down error1");
	for (int i = 0; i < Cols; i++)
	{
		//cout << "("<<point[i].x<<","<< point[i].y<<"):"<< point[i].bright << endl;
		printf("(%d,%d):%d\t, here are %d GaussPoints\n", point[i].x, point[i].y, point[i].bright,point[i].Pixnum);
	}

	/*for (int i = 0; i < Cols; i++)
	{
		free((void*)cpu_data[i]);
	}*/
	free((void*)cpu_data);
	free(cpu_data_d);
	/*for (int i = 0; i < Cols; i++)
	{
		cudaFree((void*)gpu_data[i]);
	}*/
	cudaFree(gpu_data);
	cudaFree(gpu_point);
	cudaFree(gpu_data_d);
	gpuMat.release();



}

extern "C" void GuassFitGpuHcT(Mat matImage, MPoint *point, double maxError, double minError, int yRange, int Colonce)
{

}

