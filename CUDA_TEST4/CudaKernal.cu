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

//����˷� a[M][N]*b[N][S]
cudaError_t checkCudaError(cudaError_t CudaFunction,const char* ident) {
	cudaError_t err = CudaFunction;
	if (err != cudaSuccess) {
		fprintf(stderr, "%s \t cudaError:%s\n",ident,cudaGetErrorString(cudaGetLastError()));
	}
	return err;
}

__global__ void GetGaussPointCuda(PtrStepSz<uchar1> src,MPoint *point, int **gpu_data, double maxError, double minError, int yRange, int Colonce,int Rows,int Cols) {
	int threadId = threadIdx.x;
	//printf("%d\n",threadId);
    //__shared__ int *gpu_cr;
	//gpu_cr = new int [Rows*Cols];
	//���д�������
	//MTest << <1, 1 >> > (1);
	for (int j = 0; j < Rows; j++)
	{
		for (int i = 0; i < Colonce; i++)
		{
			gpu_data[i+threadId*Colonce][j] = (int)src(j, threadId*Colonce + i).x;                
		}
	}
	//int i = 0, j = 0;
	//gpu_cr[i*Rows + j] = *((int*)&src( j, threadId*Colonce + i));
	
	//ȡÿ�����ֵλ��
	for (int i = 0; i < Colonce; i++) {
		int MaxPixel = gpu_data[i+threadId*Colonce][0];
		//printf("the first pixel is %d \n", MaxPixel);
		int MaxY = 0;
		for (int j = 1; j < Rows; j++)
		{
			if (gpu_data[i+threadId*Colonce][j] > MaxPixel)
			{
				MaxPixel = gpu_data[i + threadId * Colonce][j];
				MaxY = j;
			}
		}		

		point[threadId*Colonce + i].x = threadId * Colonce + i;
		point[threadId*Colonce + i].y = MaxY;
		point[threadId*Colonce + i].bright = MaxPixel;
	}
	
	//��˹��ɸѡ
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
		//��������
		if (Pixnum >= 3)
		{
			__shared__ int *X;
			X = new int[Pixnum * 3];
			__shared__ int *Z;
			Z = new int[Pixnum];
			//��������<3.5 ����Ƕ�ײ��к˺���
			//dim3 blockSEX(1, 0, 0);
			//dim3 threadSEX(Pixnum, 0, 0);
			//����X��Z����
			//SetElementX << <blockSEX, threadSEX >> > (gpoint, X, Pixnum);
			//����X����(n*3) Z����n*1)
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
			//��Xת��
			__shared__ int *XT;
			XT = new int[Pixnum* 3];
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < Pixnum; j++)
				{
					XT[i*Pixnum + j] = X[j * 3 + i];
				}
			}
			//��XT*X���
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
			//��SA�����
			__shared__ int *SAN;
			SAN = new int[3 * 3];


		}*/
	//}
	//delete &gpu_cr;
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
	
	}
}


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
	//�ṹ��ָ���ϴ�
	MPoint *gpu_point;
	//gpu_point = new MPoint[Cols];	
	checkCudaError(cudaMalloc((void**)&gpu_point, sizeof(MPoint)*Cols), "malloc error1");
	//�Դ�ͼ�񻺴����
	int **gpu_data;
	int *gpu_data_d;
	int **cpu_data = (int**)malloc(sizeof(int*)*Cols);
	int *cpu_data_d = (int*)malloc(sizeof(int)*Cols*Rows);
	checkCudaError(cudaMalloc((void**)&gpu_data, Cols * sizeof(int**)), "malloc error2");
	checkCudaError(cudaMalloc((void**)&gpu_data_d, Cols *Rows * sizeof(int)), " malloc error2");
	for (int i = 0; i < Cols; i++) {
		cpu_data[i] = gpu_data_d + Rows * i;
		//�׵�ַ��ֵ ��һά����תΪ��ά
	}
	checkCudaError(cudaMemcpy(gpu_data, cpu_data, sizeof(int*)*Cols, cudaMemcpyHostToDevice), "memcpy error1");
	checkCudaError(cudaMemcpy(gpu_data_d, cpu_data_d, sizeof(int)*Rows*Cols, cudaMemcpyHostToDevice), "memcpy error1");
	   



	/*if (cudaSuccess != cudaMemcpy(gpu_point, point, sizeof(point)*Cols, cudaMemcpyHostToDevice)) {
		printf("cuda memcpy up error1!\n");
	}*/
	
	//dim3 threads_all(Cols / Colonce);

	GetGaussPointCuda << <1, Cols/Colonce >> > (gpuMat, gpu_point, gpu_data, maxError, minError, yRange, Colonce, Rows, Cols);
	cudaDeviceSynchronize();
	checkCudaError(cudaMemcpy(point, gpu_point, sizeof(MPoint)*Cols, cudaMemcpyDeviceToHost), "memcpy down error1");
	for (int i = 0; i < Cols; i++)
	{
		//cout << "("<<point[i].x<<","<< point[i].y<<"):"<< point[i].bright << endl;
		printf("(%d,%d):%d\t, here are %d GaussPoints\n", point[i].x, point[i].y, point[i].bright,point[i].Pixnum);
	}

	cudaFree(gpu_point);
	cudaFree(gpu_data);
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

extern "C" void GuassFitGpuHcT(Mat matImage, MPoint *point, double maxError, double minError, int yRange, int Colonce)
{

}

