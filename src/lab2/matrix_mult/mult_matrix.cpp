#include <CL/sycl.hpp>

using  namespace  cl::sycl;

void matrix_mult_C(float *a, float *b, float *c, int N)
{
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++){
			c[i*N+j] = 0.0f;
			for(int k=0; k<N; k++)
				c[i*N+j] += a[i*N+k]*b[k*N+j]; 
		}
}

void matrix_mult_naive(sycl::queue Q, float *a, float *b, float *c, int N)
{

	// Create a command_group to issue command to the group
	Q.submit([&](handler &h) {

		// Submit the kernel
		h.parallel_for(range<2>(N, N), [=](id<2> item) {
			auto i = item[0];
			auto j = item[1];
			// CODE THAT RUNS ON DEVICE
			c[i*N+j] = 0.0f;
			for(int k=0; k<N; k++)
				c[i*N+j] += a[i*N+k]*b[k*N+j];
		}); // End of the kernel function
	}).wait();  // End of the queue commands we waint on the event reported.

}

void matrix_mult_hierarchy(sycl::queue Q, float *a, float *b, float *c, int N)
{

	const int B=16;

	// Create a command_group to issue command to the group
	Q.submit([&](handler &h) {

		range num_groups= range<2>(N/B, N/B); // N is a multiple of B
		range group_size = range<2>(B, B);
		
		/* TODO: Create a hierarchy parallelism */
		c[0] = 0.0f;

	}).wait();  // End of the queue commands we waint on the event reported.
}

void matrix_mult_local(sycl::queue Q, float *a, float *b, float *c, int N)
{

	// Local accessor, for one matrix tile:
	constexpr int tile_size = 16;


	// Create a command_group to issue command to the group
	Q.submit([&](handler &h) {

		/* TODO: Create local memory */

		/* TODO: Submit the kernel */
		c[0] = 0.0f;

	}).wait();  // End of the queue commands we waint on the event reported.

}

int check_results(float *c, float *c_test, int N)
{
	for(int i=0; i<N; i++){		
		for(int j=0; j<N; j++)
			if (abs((c_test[i*N+j]-c[i*N+j]))/c[i*N+j]>1e-4) {
				printf("  ERROR in [%i, %i]: %f %f\n", i, j, c_test[i*N+j], c[i*N+j]);
				return(0);
			}
	}
	return(1);
}

int main(int argc, char **argv) {

	if (argc!=2)  {
		std::cout << "./exec N"<< std::endl;
		return(-1);
	}

	int N = atoi(argv[1]);

	sycl::queue Q(sycl::gpu_selector{});

	std::cout << "Running on "
		<< Q.get_device().get_info<sycl::info::device::name>()
		<< std::endl;


	// a in USM
	float *a = malloc_shared<float>(N*N, Q);
	float *b = malloc_shared<float>(N*N, Q);
	float *c = malloc_shared<float>(N*N, Q);
	float *c_test = malloc_host<float>(N*N, Q);

	// Parallel for
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			a[i*N+j] = i+j;   // Init a
			b[i*N+j] = i*i+1; // Init b
		}
	}


	// Matrix Multiplication test
	matrix_mult_C(a, b, c_test, N);

	matrix_mult_naive(Q, a, b, c, N);
	if (check_results(c, c_test, N))
		printf("TEST NAIVE_MULT PASSED\n");
	else
		printf("TEST NAIVE_MULT FAILED\n");

	matrix_mult_hierarchy(Q, a, b, c, N);
	if (check_results(c, c_test, N))
		printf("TEST HIERARCHY_MULT PASSED\n");
	else
		printf("TEST HIERARCHY_MULT FAILED\n");

	matrix_mult_local(Q, a, b, c, N);
	if (check_results(c, c_test, N))
		printf("TEST LOCAL_MULT PASSED\n");
	else
		printf("TEST LOCAL_MULT FAILED\n");

	free(a, Q);
	free(b, Q);
	free(c, Q);
	free(c_test, Q);

	return 0;
}
