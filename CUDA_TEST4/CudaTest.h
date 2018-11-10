#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
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

using namespace std;

//打印cuda属性信息
int printDeviceProp(const cudaDeviceProp &prop);
//cuda初始化
bool InitCuda();
