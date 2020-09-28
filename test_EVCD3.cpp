// g++ -o test_EVD3.o test_EVD3.cpp -c -g0 -O3 -std=c++11 -I"/usr/local/cuda-10.0/include" -I/product/x86_64/local/arrayfire_3.7.2/arrayfire/include
// g++ -o evd3 test_EVD3.o -L"/usr/local/cuda-10.0/lib64" -L/product/x86_64/local/arrayfire_3.7.2/arrayfire/lib64 -lstdc++ -lafcuda -lcudart -lcublas -lcusolver
// nvcc -arch=sm_35 test_EVD3.cpp -DUSE_CACHE --ptxas-options=-v -o evd3 -std=c++11 -lstdc++ -lcudart -lcublas -lcusolver // compilation line for standalone compilation without arrayfire (need to remove all arrayfire stuffs)

#include <iostream>
#include <string>
#include <cstring>
#include <assert.h>
#include <random>
#include <algorithm>
using namespace std;

// cuda, cublas and cusolver headers
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <af/cuda.h>
#include <arrayfire.h>
using namespace af;

float *evd_standalone(const int d, const float *diag_val) {
	
	cudaError_t cu_error;
	cusolverStatus_t cusolver_status;
	cusolverDnHandle_t cusolverH;
	
	float *d_A; cudaMalloc(&d_A, sizeof(float)*(d*d));
	float *ptr_A = new float[d*d]; std::fill(ptr_A, ptr_A+(d*d), 0);
	for (int i=0; i < d; i++) ptr_A[i*d + i] = diag_val[i]; // filling diagonal of matrix with values in diag_val
	cudaMemcpy(d_A, ptr_A, sizeof(float)*(d*d), cudaMemcpyHostToDevice);
	
	float *d_D; cudaMalloc(&d_D, sizeof(float)*d);
	
	// initialize cuSolverDn
	cusolver_status = cusolverDnCreate(&cusolverH);
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors -- CUSOLVER_EIG_MODE_NOVECTOR
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	
	// compute size to allocate for temporary work array
	int lwork_db;
	cusolver_status = cusolverDnSsyevd_bufferSize(cusolverH, jobz, uplo, d, d_A, d, d_D, &lwork_db);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	printf("lwork_db = %d\n", (int)lwork_db);
	
	float *d_work; cudaMalloc(&d_work, sizeof(float)*lwork_db);
	int *d_info; cudaMalloc(&d_info, sizeof(int));
	
	// perform symmetric EVD
	cusolver_status = cusolverDnSsyevd(cusolverH, jobz, uplo, d, d_A, d, d_D, d_work, lwork_db, d_info);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	int info_disp; cudaMemcpy(&info_disp, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	if ( 0 > info_disp ) printf("Pb: %d-th parameter is wrong\n", -info_disp);
	cu_error = cudaDeviceSynchronize();
	assert(cudaSuccess==cu_error);
	
	float *temp_D = new float[d]; cudaMemcpy(temp_D, d_D, sizeof(float)*d, cudaMemcpyDeviceToHost);
	
	return temp_D;
}

af::array evd_interop(const int d, float *diag_val) {
	
	cudaError_t cu_error;
	cusolverStatus_t cusolver_status;
	cusolverDnHandle_t cusolverH;
	
	// creates diagonal matrix of size d x d (diagonal with positive values: so it is symmetric hermitian)
	af::array A = diag(af::array(d, diag_val), 0, false); A.eval();
	af::array D = constant(0.f, d); D.eval(); // D will contain the eigenvalues
	float *d_A = A.device<float>();
	float *d_D = D.device<float>();
	
	int cuda_id = afcu::getNativeId(getDevice());
	printf("cuda_id = %d\n", cuda_id);
	cudaStream_t af_cuda_stream = afcu::getStream(cuda_id);
	
	// initialize cuSolverDn
	cusolver_status = cusolverDnCreate(&cusolverH);
	cusolver_status = cusolverDnSetStream(cusolverH, af_cuda_stream);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors -- CUSOLVER_EIG_MODE_NOVECTOR
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	
	// compute size to allocate for temporary work array
	int lwork_db;
	cusolver_status = cusolverDnSsyevd_bufferSize(cusolverH, jobz, uplo, d, d_A, d, d_D, &lwork_db);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	printf("lwork_db = %d\n", (int)lwork_db);
	
	float *d_work; cudaMalloc (( void **)& d_work , sizeof ( float )* lwork_db );
	int *d_info; cudaMalloc((void**)& d_info, sizeof(int ));
	
	// perform symmetric EVD
	cusolver_status = cusolverDnSsyevd(cusolverH, jobz, uplo, d, d_A, d, d_D, d_work, lwork_db, d_info);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	int info_disp; cudaMemcpy(&info_disp, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	if ( 0 > info_disp ) printf("Pb: %d-th parameter is wrong\n", -info_disp);
	cu_error = cudaDeviceSynchronize();
	assert(cudaSuccess==cu_error);
	
	D.unlock();
	//~ af::array new_D = af::array(d, d_D, afDevice); new_D.eval(); // arrayfire is taking back control of D
	af::sync();
	
	return D;
	
}

int main(int argc, char *argv[]) {
	
	int device = 0;
	try {
		af::setDevice(device);
		af::info();
	} catch (af::exception &ae) {
		std::cerr << ae.what() << std::endl;
		return 0;
	}
	
	std::random_device rd; std::mt19937 generator(rd());
	std::uniform_real_distribution<float> runif(1.f, 10.f);
	
	const int d = 300; // dimensionality of matrix to perform EVD on
	
	// generate strictly positive values for diagonal matrix
	float *diag_val = new float[d]; for (int i=0; i < d; i++) diag_val[i] = runif(generator);
	sort(diag_val, diag_val+d);
	for (int i=0; i < 10; i++) printf("%f | ", diag_val[i]);
	
	// af::array D = evd_interop(d, diag_val); // using cuda-arrayfire interoperability
	
	float *ptr_D = evd_standalone(d, diag_val); // computing evd in a standalone way (not using arrayfire in any way)
	for (int i=0; i < 10; i++) printf("%f | ", ptr_D[i]);
	printf("\n");
	af::array D = af::array(d, ptr_D);
	
	af::array af_trace = sum(D, 0); // failing here
	float trace_f; af_trace.host(&trace_f);
	printf("trace = %f\n", trace_f);
	af_print(D(af::seq(0, 10)));
	
	return 1;
	
}
