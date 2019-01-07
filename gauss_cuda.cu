#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include <string.h>

#define MAXBLOCKSIZE 512

int Size;
float *a, *b, *FinalArray;
float *m;


void ForwardFunction();
void BackwardSubstitution();
//void MultiplierMatrix(float *m, float *a, int Size, int t);
//void ForwardEliminate(float *m, float *a, float *b,int Size, int j1, int t);
void InitializeMatrix(float *ARR, int nrow, int ncol);
void InitializeArray(float *ARR, int ARR_size);
void PrintMatrix(float *ARR, int nrow, int ncolumn);
void PrintArray(float *ARR, int ARR_size);
void checkCUDAError(const char *msg);

unsigned int totalKernelTime = 0;

int main(int argc, char *argv[])
{
    int verbose = 0;
    int i = 0;
	printf("The Size of input matrix a is:\n");
	scanf("%d",&Size);
	
	a = (float *) malloc(Size * Size * sizeof(float));
	 
	InitializeMatrix(a, Size, Size);
	//printf("The input matrix a is:\n");
	//PrintMatrix(a, Size, Size);
	b = (float *) malloc(Size * sizeof(float));
	
	InitializeArray(b, Size);
	//printf("The input array b is:\n");
	//PrintArray(b, Size);
		
	m = (float *) malloc(Size * Size * sizeof(float));
    
	for (i=0; i<Size*Size; i++)
		*(m+i) = 0.0;
	
	//begin timing
    struct timeval time_start;
    gettimeofday(&time_start, NULL);	
    
    // run kernels
    ForwardFunction();
    
    //end timing
    struct timeval time_end;
    gettimeofday(&time_end, NULL);
    unsigned int time_total = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);
    
    if (verbose) {
        printf("Matrix m is: \n");
        PrintMatrix(m, Size, Size);

        printf("Matrix a is: \n");
        PrintMatrix(a, Size, Size);

        printf("Array b is: \n");
        PrintArray(b, Size);
    }
    BackwardSubstitution();
    if (verbose) {
        printf("The final solution is: \n");
        PrintArray(FinalArray,Size);
    }
    printf("\nTime total (including memory transfers)\t%f sec\n", time_total * 1e-7);
    printf("Time for CUDA kernels:\t%f sec\n",totalKernelTime * 1e-7);
    
    free(m);
    free(a);
    free(b);
}
 
__global__ void MultiplierMatrix(float *m_cuda, float *a_cuda, int Size, int t)
{   
	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	*(m_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);
}

/*-------------------------------------------------------
 ** ForwardEliminate() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */ 

__global__ void ForwardEliminate(float *m_cuda, float *a_cuda, float *b_cuda,int Size, int j1, int t)
{
	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	if(threadIdx.y + blockIdx.y * blockDim.y >= Size-t) return;
	
	//*(m_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);
	int xidx = blockIdx.x * blockDim.x + threadIdx.x;
	int yidx = blockIdx.y * blockDim.y + threadIdx.y;
	
	a_cuda[Size*(xidx+1+t)+(yidx+t)] -= m_cuda[Size*(xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];
	if(yidx == 0){
		b_cuda[xidx+1+t] -= m_cuda[Size*(xidx+1+t)+(yidx+t)] * b_cuda[t];
	}
}

/*------------------------------------------------------
 ** ForwardFunction() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardFunction()
{
	int t;
    float *m_cuda,*a_cuda,*b_cuda;
	
	// allocate memory on GPU
	cudaMalloc((void **) &m_cuda, Size * Size * sizeof(float));
	 
	cudaMalloc((void **) &a_cuda, Size * Size * sizeof(float));
	
	cudaMalloc((void **) &b_cuda, Size * sizeof(float));	

	// copy memory to GPU
	cudaMemcpy(m_cuda, m, Size * Size * sizeof(float),cudaMemcpyHostToDevice );
	cudaMemcpy(a_cuda, a, Size * Size * sizeof(float),cudaMemcpyHostToDevice );
	cudaMemcpy(b_cuda, b, Size * sizeof(float),cudaMemcpyHostToDevice );
	
	int block_size,grid_size;
	
	block_size = MAXBLOCKSIZE;
	grid_size = (Size/block_size) + (!(Size%block_size)? 0:1);
	//printf("1d grid size: %d\n",grid_size);


	dim3 dimBlock(block_size);
	dim3 dimGrid(grid_size);
	//dim3 dimGrid( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );
	
	int blockSize2d, gridSize2d;
	blockSize2d = 4;
	gridSize2d = (Size/blockSize2d) + (!(Size%blockSize2d?0:1)); 
	
	dim3 dimBlockXY(blockSize2d,blockSize2d);
	dim3 dimGridXY(gridSize2d,gridSize2d);

    // begin timing kernels
    struct timeval time_start;
    gettimeofday(&time_start, NULL);
	for (t=0; t<(Size-1); t++) {
		MultiplierMatrix<<<dimGrid,dimBlock>>>(m_cuda,a_cuda,Size,t);
		cudaThreadSynchronize();
		ForwardEliminate<<<dimGridXY,dimBlockXY>>>(m_cuda,a_cuda,b_cuda,Size,Size-t,t);
		cudaThreadSynchronize();
		checkCUDAError("ForwardEliminate");
	}
	// end timing kernels
	struct timeval time_end;
    gettimeofday(&time_end, NULL);
    totalKernelTime = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);
	
	// copy memory back to CPU
	cudaMemcpy(m, m_cuda, Size * Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaMemcpy(a, a_cuda, Size * Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaMemcpy(b, b_cuda, Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaFree(m_cuda);
	cudaFree(a_cuda);
	cudaFree(b_cuda);
}

/*------------------------------------------------------
 ** BackwardSubstitution() -- Backward substitution
 **------------------------------------------------------
 */

void BackwardSubstitution()
{
	// create a new vector to hold the final answer
	FinalArray = (float *) malloc(Size * sizeof(float));
	int i,j;
	for(i=0;i<Size;i++){
		FinalArray[Size-i-1]=b[Size-i-1];
		for(j=0;j<i;j++)
		{
			FinalArray[Size-i-1]-=*(a+Size*(Size-i-1)+(Size-j-1)) * FinalArray[Size-j-1];
		}
		FinalArray[Size-i-1]=FinalArray[Size-i-1]/ *(a+Size*(Size-i-1)+(Size-i-1));
	}
}

void InitializeMatrix(float *ARR, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			ARR[i*Size+j] = (float)rand()/(float)(RAND_MAX/10);
		}
	}  
}

/*------------------------------------------------------
 ** PrintMatrix() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMatrix(float *ARR, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			printf("%8.2f ", *(ARR+Size*i+j));
		}
		printf("\n");
	}
	printf("\n");
}

/*------------------------------------------------------
 ** InitializeArray() -- Initialize the array (vector) with random
 ** data
 **------------------------------------------------------
 */
void InitializeArray(float *ARR, int ARR_size)
{
	int i;
	
	for (i=0; i<ARR_size; i++) {
		ARR[i] = (float)rand()/(float)(RAND_MAX/10);
	}
}  

/*------------------------------------------------------
 ** PrintArray() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintArray(float *ARR, int ARR_size)
{
	int i;
	for (i=0; i<ARR_size; i++) {
		printf("%.2f ", ARR[i]);
	}
	printf("\n\n");
}
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        printf("Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}
