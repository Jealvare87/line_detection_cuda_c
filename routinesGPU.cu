#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"

#define DEG2RAD 0.017453f
#define NTHREADS 32
#define SHARED_S 180

/*--Global Variables--*/

__constant__ float sin_const[SHARED_S];
__constant__ float cos_const[SHARED_S];

/*--Canny--*/

__device__ void check_neigh(float *G, uint8_t *image_out, float hithres, int width, int i, int j, int ii, int jj){
		if(ii > -2 && ii < 2){
			if(jj > -2 && jj < 2){
				if (G[(i+ii)*width+j+jj]>hithres){
					image_out[i*width+j] = 255;
					jj++;
					check_neigh(G, image_out, hithres, width, i, j, ii, jj);
				}
			}	
			else{
				ii++;
				jj = -1;
				check_neigh(G, image_out, hithres, width, i, j, ii, jj);
			}
		}
}


__global__ void noise_reduc(uint8_t *im, float *NR, int height, int width){
	
	int i, j, posi, posj, tam_max;

	i = blockIdx.y * blockDim.y + threadIdx.y+2;
	j = blockIdx.x * blockDim.x + threadIdx.x+2;

	__shared__ uint8_t im_share[NTHREADS+4][NTHREADS+4];

	posj = threadIdx.y+2;
	posi = threadIdx.x+2;

	tam_max = NTHREADS;
	
	/* Shared Cases */

	if(posi > 1 && posi < 4){							
		im_share[posi-2][posj] = im[(i-2)*width+j];
		if(posj > 1 && posi < 4){											// Up-Left Corner
			im_share[posi][posj-2] = im[i*width+j-2];
			im_share[posi-2][posj-2] = im[(i-2)*width+j-2];
		}
		else if(posj > tam_max-1 && posj < tam_max+2){						// Up-Right Corner
			im_share[posi][posj+2] = im[i*width+j+2];
			im_share[posi-2][posj+2] = im[(i-2)*width+j+2];
		}
	}
	else if(posi > tam_max-1 && posi < tam_max+2){
		im_share[posi+2][posj] = im[(i+2)*width+j];
		if(posj >= 2 && posj < 4){											// Down-Left Corner
			im_share[posi][posj-2] = im[i*width+j-2];
			im_share[posi+2][posj-2] = im[(i+2)*width+j-2];
		}
		else if(posj > tam_max-1 && posj < tam_max+2){						// Down-Right Corner
			im_share[posi][posj+2] = im[i*width+j+2];
			im_share[posi-2][posj+2] = im[(i+2)*width+j+2];
		}
	}
	else if(posj > 1 && posj < 4){											// Left Border
		im_share[posi][posj-2] = im[i*width+j-2];
	}
	else if(posj > tam_max-1 && posj < tam_max+2){							// Right Border
		im_share[posi][posj+2] = im[i*width+j+2];
	}

	im_share[posi][posj] = im[i*width+j];


	__syncthreads();

	/* Convolution */
	/*
	if ((i >= 2 && i < (height-2)) &&
		(j >= 2 && j < (width-2))){
			NR[i*width+j] =
				 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
				+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
				+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
				+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
				+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
				/159.0;
			__syncthreads();

	}*/
	
	if ((i > 1 && i < height-2) &&
		(j > 1 && j < width-2)){
			NR[i*width+j] =
				 (2.0*im_share[posi-2][posj-2] +  4.0*im_share[posi-2][posj-1] +  5.0*im_share[posi-2][posj] +  4.0*im_share[posi-2][posj+1] + 2.0*im_share[posi-2][posj+2]
				+ 4.0*im_share[posi-1][posj-2] +  9.0*im_share[posi-1][posj-1] + 12.0*im_share[posi-1][posj] +  9.0*im_share[posi-1][posj+1] + 4.0*im_share[posi-1][posj+2]
				+ 5.0*im_share[posi][posj-2] + 12.0*im_share[posi][posj-1] + 15.0*im_share[posi][posj] + 12.0*im_share[posi][posj+1] + 5.0*im_share[posi][posj+2]
				+ 4.0*im_share[posi+1][posj-2] +  9.0*im_share[posi+1][posj-1] + 12.0*im_share[posi+1][posj] +  9.0*im_share[posi+1][posj+1] + 4.0*im_share[posi+1][posj+2]
				+ 2.0*im_share[posi+2][posj-2] +  4.0*im_share[posi+2][posj-1] +  5.0*im_share[posi+2][posj] +  4.0*im_share[posi+2][posj+1] + 2.0*im_share[posi+2][posj+2])
				/159.0;
	}
}


__global__ void inten_grad(float *NR, float *G, float *phi, int height, int width){
	int i, j, posi, posj, tam_max;
	float PI = 3.141593;
	float Gx, Gy;

	i = blockIdx.y * blockDim.y + threadIdx.y+2;
	j = blockIdx.x * blockDim.x + threadIdx.x+2;

	__shared__ float nrs[NTHREADS+4][NTHREADS+4];

	posi = threadIdx.y+2;
	posj = threadIdx.x+2;

	tam_max = NTHREADS;
	
	/* Shared Cases */
	if(posi > 1 && posi < 4){							
		nrs[posi-2][posj] = NR[(i-2)*width+j];
		if(posj > 1 && posi < 4){											// Up-Left Corner
			nrs[posi][posj-2] = NR[i*width+j-2];
			nrs[posi-2][posj-2] = NR[(i-2)*width+j-2];
		}
		else if(posj > tam_max-1 && posj < tam_max+2){						// Up-Right Corner
			nrs[posi][posj+2] = NR[i*width+j+2];
			nrs[posi-2][posj+2] = NR[(i-2)*width+j+2];
		}
	}
	else if(posi > tam_max-1 && posi < tam_max+2){
		nrs[posi+2][posj] = NR[(i+2)*width+j];
		if(posj >= 2 && posj < 4){											// Down-Left Corner
			nrs[posi][posj-2] = NR[i*width+j-2];
			nrs[posi+2][posj-2] = NR[(i+2)*width+j-2];
		}
		else if(posj > tam_max-1 && posj < tam_max+2){						// Down-Right Corner
			nrs[posi][posj+2] = NR[i*width+j+2];
			nrs[posi-2][posj+2] = NR[(i+2)*width+j+2];
		}
	}
	else if(posj > 1 && posj < 4){											// Left Border
		nrs[posi][posj-2] = NR[i*width+j-2];
	}
	else if(posj > tam_max-1 && posj < tam_max+2){							// Right Border
		nrs[posi][posj+2] = NR[i*width+j+2];
	}

	nrs[posi][posj] = NR[i*width+j];

	__syncthreads();


	// Intensity gradient of the image
	if ((i >= 2 && i < (height-2)) &&
		(j >= 2 && j < (width-2))){
			Gx = 
				 (1.0*nrs[posi-2][posj-2] +  2.0*nrs[posi-2][posj-1] +  (-2.0)*nrs[posi-2][posj+1] + (-1.0)*nrs[posi-2][posj+2]
				+ 4.0*nrs[posi-1][posj-2] +  8.0*nrs[posi-1][posj-1] +  (-8.0)*nrs[posi-1][posj+1]+ (-4.0)*nrs[posi-1][posj+2]
				+ 6.0*nrs[posi  ][posj-2] + 12.0*nrs[posi  ][posj-1] + (-12.0)*nrs[posi  ][posj+1] + (-6.0)*nrs[posi ][posj+2]
				+ 4.0*nrs[posi+1][posj-2] +  8.0*nrs[posi+1][posj-1] +  (-8.0)*nrs[posi+1][posj+1]+ (-4.0)*nrs[posi+1][posj+2]
				+ 1.0*nrs[posi+2][posj-2] +  2.0*nrs[posi+2][posj-1] +  (-2.0)*nrs[posi+2][posj+1] + (-1.0)*nrs[posi+2][posj+2]);


			Gy = 
				 ((-1.0)*nrs[posi-2][posj-2] + (-4.0)*nrs[posi-2][posj-1] +  (-6.0)*nrs[posi-2][posj] + (-4.0)*nrs[posi-2][posj+1] + (-1.0)*nrs[posi-2][posj+2]
				+ (-2.0)*nrs[posi-1][posj-2] + (-8.0)*nrs[posi-1][posj-1] + (-12.0)*nrs[posi-1][posj] + (-8.0)*nrs[posi-1][posj+1]+ (-2.0)*nrs[posi-1][posj+2]
				+    2.0*nrs[posi+1][posj-2] +    8.0*nrs[posi+1][posj-1] +    12.0*nrs[posi+1][posj] +    8.0*nrs[posi][posj+1] +    2.0*nrs[posi+1][posj+2]
				+    1.0*nrs[posi+2][posj-2] +    4.0*nrs[posi+2][posj-1] +     6.0*nrs[posi+2][posj] +    4.0*nrs[posi+2][posj+1] +    1.0*nrs[posi+2][posj+2]);
			
			G[i*width+j]   = sqrtf((Gx*Gx)+(Gy*Gy));	//G = √Gx²+Gy²
			phi[i*width+j] = atan2f(fabsf(Gy),fabsf(Gx));

			if(fabsf(phi[i*width+j])<=PI/8 )
				phi[i*width+j] = 0;
			else if (fabsf(phi[i*width+j])<= 3*(PI/8))
				phi[i*width+j] = 45;
			else if (fabsf(phi[i*width+j]) <= 5*(PI/8))
				phi[i*width+j] = 90;
			else if (fabsf(phi[i*width+j]) <= 7*(PI/8))
				phi[i*width+j] = 135;
			else phi[i*width+j] = 0;
	}
	__syncthreads();
}

__global__ void edge_hysterisis(uint8_t *image_out, float *phi, float *G, float lowthres, float hithres, int height, int width){
	int i, j;
	uint8_t pedge;

	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i >= 3 && i < (height-3)) &&
		(j >= 3 && j < (width-3))){
			pedge = 0;
			if(phi[i*width+j] == 0){
				if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
					pedge = 1;

			} else if(phi[i*width+j] == 45) {
				if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
					pedge = 1;

			} else if(phi[i*width+j] == 90) {
				if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
					pedge = 1;

			} else if(phi[i*width+j] == 135) {
				if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
					pedge = 1;
			}
	
	__syncthreads();

		image_out[i*width+j] = 0;
		if(G[i*width+j]>hithres && pedge)
			image_out[i*width+j] = 255;
		else if(pedge && G[i*width+j]>=lowthres && G[i*width+j]<hithres){
			// check neighbours 3x3
			check_neigh(G, image_out, hithres, width, i, j, -1, -1); //Recursive
		}
	}
	__syncthreads();

}


void canny(uint8_t *im, uint8_t *dImageOut,
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level,
	int height, int width)
{
	float lowthres, hithres;

	/*CUDA Attributes*/
	uint8_t *dImageIn;
	float *dNR, *dG, *dphi;

	/*CUDA Mallocs*/
	cudaMalloc((void **)&dImageIn, height*width*sizeof(uint8_t));	//im
	cudaMalloc((void **)&dNR, height*width*sizeof(float));			//NR
	cudaMalloc((void **)&dG, height*width*sizeof(float));			//G
	cudaMalloc((void **)&dphi, height*width*sizeof(float));			//phi

	cudaMemcpy(dImageIn, im, height*width*sizeof(uint8_t), cudaMemcpyHostToDevice);

	/*Define blocks and threads*/
	dim3 dimBlock(NTHREADS,NTHREADS);
	int blocks_width = width/NTHREADS;
	int blocks_height = height/NTHREADS;
	/*Se mira por separado ya que la imagen
	  puede o no ser cuadrada*/
	if (width%NTHREADS>0) blocks_width++;
	if (height%NTHREADS>0) blocks_height++;

	dim3 dimGrid(blocks_width, blocks_height);

	noise_reduc<<<dimGrid, dimBlock>>>(dImageIn, dNR, height, width);
	cudaDeviceSynchronize();
	inten_grad<<<dimGrid, dimBlock>>>(dNR, dG, dphi, height, width);
	cudaDeviceSynchronize();
	
	// Hysteresis Thresholding
		lowthres = level/2;
		hithres  = 2*(level);
	
	edge_hysterisis<<<dimGrid, dimBlock>>>(dImageOut, dphi, dG, lowthres, hithres, height, width);
	cudaDeviceSynchronize();

	cudaFree(dphi);
	cudaFree(dG);
	cudaFree(dNR);
	cudaFree(dImageIn);
}


void getlines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height, 
	float *sin_table, float *cos_table,
	int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	int rho, theta;
	uint32_t max;

	for(rho=0;rho<accu_height;rho++)
	{
		for(theta=0;theta<accu_width;theta++)  
		{  

			if(accumulators[(rho*accu_width) + theta] >= threshold)  
			{  
				//Is this point a local maxima (9x9)  
				max = accumulators[(rho*accu_width) + theta]; 
				for(int ii=-4;ii<=4;ii++)  
				{  
					for(int jj=-4;jj<=4;jj++)  
					{  
						if( (ii+rho>=0 && ii+rho<accu_height) && (jj+theta>=0 && jj+theta<accu_width) )  
						{  
							if( accumulators[((rho+ii)*accu_width) + (theta+jj)] > max )  
							{
								max = accumulators[((rho+ii)*accu_width) + (theta+jj)];
							}  
						}  
					}  
				}  

				if(max == accumulators[(rho*accu_width) + theta]) //local maxima
				{
					int x1, y1, x2, y2;  
					x1 = y1 = x2 = y2 = 0;  

					if(theta >= 45 && theta <= 135)  
					{
						if (theta>90) {
							//y = (r - x cos(t)) / sin(t)  
							x1 = width/2;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);  
						} else {
							//y = (r - x cos(t)) / sin(t)  
							x1 = 0;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width*2/5;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2); 
						}
					} else {
						//x = (r - y sin(t)) / cos(t);  
						y1 = 0;  
						x1 = ((float)(rho-(accu_height/2)) - ((y1 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
						y2 = height;  
						x2 = ((float)(rho-(accu_height/2)) - ((y2 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
					}
					x1_lines[*lines] = x1;
					y1_lines[*lines] = y1;
					x2_lines[*lines] = x2;
					y2_lines[*lines] = y2;
					(*lines)++;
				}
			}
		}
	}
}

void init_cos_sin_table(float *sin_table, float *cos_table, int n)
{
	int i;
	for (i=0; i<n; i++)
	{
		sin_table[i] = sinf(i*DEG2RAD);
		cos_table[i] = cosf(i*DEG2RAD);
	}
}

__global__ void hough_gpu(uint8_t *im, uint32_t *accumulators, float center_x, float center_y, 
	int height, int width , float hough_h){

	int i, j, theta;

	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.x * blockDim.x + threadIdx.x;


	if((i >= 0 && i < height)&&
		(j >= 0 && j < width)){
		if( im[ (i*width) + j] > 250 ) // Pixel is edge  
		{  
			for(theta=0;theta<180;theta++)  
			{  
				float rho = ( ((float)j - center_x) * cos_const[theta]) + (((float)i - center_y) * sin_const[theta]);
				//Atomic Function
				atomicAdd(&accumulators[ (int)((round(rho + hough_h) * 180.0)) + theta], 1);
			}
		} 
	} 
	__syncthreads();

}

void houghtransform(uint8_t *dim, int width, int height, uint32_t *accumulators, int accu_width, int accu_height, 
	float *sin_table, float *cos_table)
{
	int i;

	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);

	for(i=0; i<accu_width*accu_height; i++)
		accumulators[i]=0;	
	
	float center_x = width/2.0; 
	float center_y = height/2.0;
	/*CUDA Variables*/
	uint32_t *dacumm;

	/*Define blocks and threads*/
	/*2D*/
	dim3 dimBlock(NTHREADS,NTHREADS);
	int blocks_width = width/NTHREADS;
	int blocks_height = height/NTHREADS;
	if (width%NTHREADS>0) blocks_width++;
	if (height%NTHREADS>0) blocks_height++;

	dim3 dimGrid(blocks_width, blocks_height);

	/*CUDA Mallocs*/
	cudaMalloc((void **)&dacumm, accu_width*accu_height*sizeof(uint32_t));
	cudaMemcpy(dacumm, accumulators, accu_width*accu_height*sizeof(uint32_t), cudaMemcpyHostToDevice);
	/* Constant Copies */
	cudaMemcpyToSymbol(sin_const, sin_table, 180*sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cos_const, cos_table, 180*sizeof(float), 0, cudaMemcpyHostToDevice);

	hough_gpu<<<dimGrid, dimBlock>>>(dim, dacumm, center_x, center_y, height, width, hough_h);
	cudaDeviceSynchronize();
	cudaMemcpy(accumulators, dacumm, accu_width*accu_height*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	
	cudaFree(dacumm);
}

// Line Asist GPU

void line_asist_GPU(uint8_t *im, int height, int width,
	uint8_t *imEdge, float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float *sin_table, float *cos_table, 
	uint32_t *accum, int accu_height, int accu_width,
	int *x1, int *x2, int *y1, int *y2, int *nlines)
{

	/* To do */
		int threshold;
		uint8_t *dImageOut;


	/* CUDA MALLOCS */
	cudaMalloc((void **)&dImageOut, height*width*sizeof(uint8_t));	//image_out

	/* Canny */
	canny(im, dImageOut,
		NR, G, phi, Gx, Gy, pedge,
		1000.0f, //level
		height, width);
	cudaDeviceSynchronize();

	/* Hough transform */
	houghtransform(dImageOut, width, height, accum, accu_width, accu_height, sin_table, cos_table);
	cudaDeviceSynchronize();

	if (width>height) threshold = width/6;
	else threshold = height/6;

	/* No transformar */
	getlines(threshold, accum, accu_width, accu_height, width, height, 
		sin_table, cos_table,
		x1, y1, x2, y2, nlines);

	/* CUDA Free */
	cudaFree(dImageOut);
}