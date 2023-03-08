#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

//CUDA
#include <cuda.h>

double wtime(void)
{
        static struct timeval   tv0;
        double time_;

        gettimeofday(&tv0,(struct timezone*)0);
        time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
        return( time_/1000000);
}


void addMatrix(float *a, float *b, float *c, int N)
{
	int i, j, idx;
	for (i=0; i<N; i++)
		for(j=0; j<N;j++){
			idx = i*N+j;
			a[idx]=b[idx]+c[idx];
		}
} 


__global__ void addMatrixGPU(float *a, float *b, float *c, int N )
{
	....
}

int main(int argc, char *argv[])
{
	float *a, *b, *c, *a_host;
	float *a_GPU, *b_GPU, *c_GPU;

	int i, j, N;

	double t0, t1;


	if(argc>1) {
		N = atoi(argv[1]); printf("N=%i\n", N);
	} else {
		printf("Error!!!! \n ./exec number\n");
	return (0);
	}

	// Mallocs CPU
	a  = (float *)malloc(sizeof(float)*N*N);
	b  = (float *)malloc(sizeof(float)*N*N);
	c  = (float *)malloc(sizeof(float)*N*N);
	for (i=0; i<N*N; i++){ b[i] = i-1; c[i] = i;}

	/*****************/
	/* Add Matrix CPU*/
	/*****************/
	t0 = wtime();
	addMatrix(a, b, c, N);
	t1 = wtime(); printf("Time CPU=%f\n", t1-t0);

	/* Mallocs GPU */
	cudaMalloc(...);
	cudaMalloc(...);
	cudaMalloc(...);

	/* CPU->GPU */
	cudaMemcpy(...);
	cudaMemcpy(...);

	/*****************/
	/* Add Matrix GPU*/
	/*****************/
	dim3 dimBlock(...,...);
	dim3 dimGrid(...,...);
	t0 = wtime();
	addMatrixGPU<<<dimGrid,dimBlock>>>(a_GPU, b_GPU, c_GPU, N);
	cudaThreadSynchronize();
	t1 = wtime(); printf("Time GPU=%f\n", t1-t0);

	/* GPU->CPU */
	a_host  = (float *)malloc(sizeof(float)*N*N);
	cudaMemcpy(...);

	/************/
	/* Results  */
	/************/
	for (i=0; i<N; i++)
		for (j=0; j<N; j++)
			if(fabs(a[i*N+j]-a_host[i*N+j])>1e-5){
				printf("a!=a_host in (%i,%i): ", i,j);
				printf("A[%i][%i] = %f A_GPU[%i][%i]=%f\n", i, j, a[i*N+j], i, j, a_host[i*N+j] );
			}

	/* Free CPU */
	free(a);
	free(b);
	free(c);
	free(a_host);

	/* Free GPU */
	cudaFree(...);
	cudaFree(...);
	cudaFree(...);


	return(1);
}

