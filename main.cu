#include <iostream>
#include <ctime>

__global__ void init_primes_kernel(int *prime, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	while(index + offset < n){
		prime[index + offset] = index + offset + 1;

		offset += stride;
	}
}


__global__ void sieve_of_eratosthenes_kernel(int *prime, unsigned int n, unsigned int sqrRootN)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x + 2;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	while(index + offset < sqrRootN){
		unsigned int temp = index + offset;
		for(unsigned int i=temp*temp;i<n;i+=temp){
			prime[i-1] = 0;
		}

		offset += stride;
	}

}


int main()
{
	unsigned int N = 1*100*1024*1000;
	unsigned int M = (unsigned int)sqrt(N);
	int *h_primes;
	int *d_primes;


	// allocate memory
	h_primes = (int*)malloc(N*sizeof(int));
	cudaMalloc((void**)&d_primes, N*sizeof(int));


	// timing on gpu
	float gpu_elapsed_time;
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start, 0);


	// call kernel
	dim3 gridSize = 32;
	dim3 blockSize = 32;
	init_primes_kernel<<< gridSize, blockSize >>>(d_primes, N);
	sieve_of_eratosthenes_kernel<<< gridSize, blockSize >>>(d_primes, N, M);


	// copy results back to host
	cudaMemcpy(h_primes, d_primes, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);
	std::cout<<"GPU took: "<<gpu_elapsed_time<<std::endl;

	// cpu version
	for(unsigned int i=0;i<N;i++){
		h_primes[i] = i+1;
	}
	clock_t cpu_start = clock();
	for(unsigned int i=0;i<M;i++){
		unsigned int start = (i+2)*(i+2);
		for(unsigned int j=start;j<N;j+=(i+2)){
			h_primes[j-1] = 0;
		}
	}
	clock_t cpu_stop = clock();
	clock_t cpu_elapsed_time = 1000*(cpu_stop - cpu_start)/CLOCKS_PER_SEC;
	std::cout<<"The cpu took: "<<cpu_elapsed_time<<" milli-seconds"<<std::endl;

	 for(unsigned int i=0;i<N;i++){
	 	if(h_primes[i] != 0){
	 		std::cout<<h_primes[i]<<"  ";
	 	}
	 }
	 std::cout<<""<<std::endl;


	// free memory
	free(h_primes);
	cudaFree(d_primes);
}
