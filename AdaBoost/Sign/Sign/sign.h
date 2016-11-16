#pragma once 

extern "C" __declspec(dllexport) int find_min_error(int* X,int rows,int cols,int* y,double* w,double& min_error,int& is_less,int& feature_index);