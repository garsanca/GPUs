#include <CL/sycl.hpp>

using  namespace  cl::sycl;


void dummy(sycl::queue &Q, int N) {

	// Create a kernel to perform c=a+b
	Q.submit([&](handler &h) { 
		// Submit the kernel
		h.single_task([=]() {
			int tmp = N;
		});
	}).wait();       // End of the queue commands we waint on the event reported.
}

void MatrixAdd1(sycl::queue &Q, int *a, int *b,
	int *c, int N, int iter) {

	// Create a kernel to perform c=a+b
	Q.submit([&](handler &h) { 
		// Submit the kernel
		h.parallel_for(range<2>(N, N), [=](id<2> item) {
			auto i = item[?];
			auto j = item[?];
			for (int it = 0; it < iter; it++)
				c[i*N+j] = a[i*N+j] + b[i*N+j];
		});
	}).wait();
}

void MatrixAdd2(sycl::queue &Q, int *a, int *b,
	int *c, int N, int iter) {

	// Create a kernel to perform c=a+b
	Q.submit([&](handler &h) { 
		// Submit the kernel
		h.parallel_for(range<2>(N, N), [=](id<2> item) {
			auto i = item[0];
			auto j = item[1];
			for (int it = 0; it < iter; it++)
				c[?*N+?] = a[?*N+?] + b[?*N+?];
		});
	}).wait();
}


int test(int *c_test, int *c, int N)
{
	for (int i=0; i<N; i++)
		if (c_test[i]!=c[i])
			return(0);

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


	int *a = malloc_shared<int>(N*N, Q);
	int *b = malloc_shared<int>(N*N, Q);
	int *c = malloc_shared<int>(N*N, Q);
	int *c_gt = static_cast<int *>(malloc(N*N * sizeof(int)));

	// Parallel for
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++){
			a[i*N+j] = j<<(i+1);   // Init a
			b[i*N+j] = i*i;        // Init b
			c_gt[i*N+j] = a[i*N+j] + b[i*N+j];
		}

	dummy(Q, N);

	/* benchmarking */
	auto start = std::chrono::steady_clock::now();
	MatrixAdd1(Q, a, b, c, 100, N);
	auto end = std::chrono::steady_clock::now();
	printf("Time VectorAdd1=%ld usecs\n", (end - start).count());

	start = std::chrono::steady_clock::now();
	MatrixAdd2(Q, a, b, c, 100, N);
	end = std::chrono::steady_clock::now();
	printf("Time VectorAdd2=%ld usecs\n", (end - start).count());

	free(a, Q);
	free(b, Q);
	free(c, Q);
	free(c_gt);

	return 0;
}
