#include <iostream>
using namespace std; 
#include "sign.h" 


int find_min_error(int* X,int rows,int cols,int* y,double* w,double& min_error,int& is_less,int& feature_index)
{
	int index = -1;

	for(int row=0;row<rows;row++)
	{
		for(int i=0;i<3;i++)
		{
			double score = 0;

			for(int j=0;j<cols;j++)
			{
				int val = -1;
				if(X[row*cols+j]<i)
				{
					val = 1;
				}

				if(val*y[j]<0)
					score += w[j];

			}

			if(score < min_error)
			{
				index = i;
				is_less = 1;
				min_error = score;
				feature_index = row;
			}
		}
	}

	for(int row=0;row<rows;row++)
	{
		for(int i=0;i<3;i++)
		{
			double score = 0;

			for(int j=0;j<cols;j++)
			{
				int val = 1;
				if(X[row*cols+j]<i)
				{
					val = -1;
				}

				if(val*y[j]<0)
					score += w[j];

			}

			if(score < min_error)
			{
				index = i;
				is_less = 0;
				min_error = score;
				feature_index = row;
			}
		}
	}

	return index;
}