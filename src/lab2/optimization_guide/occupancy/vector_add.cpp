#include <sycl/sycl.hpp>

using  namespace  sycl;


void dummy(sycl::queue &Q, int N) {

	// Create a kernel to perform c=a+b
	Q.submit([&](handler &h) { 
		// Submit the kernel
		h.single_task([=]() {
			int tmp = N;
		});
	}).wait();       // End of the queue commands we waint on the event reported.
}

void VectorAdd1(sycl::queue &Q, int *a, int *b,
	int *c, int N, int iter) {

	// Create a kernel to perform c=a+b
	Q.submit([&](handler &h) { 
		// Submit the kernel
		h.parallel_for(N, [=](id<1> i) {
			for (int it = 0; it < iter; it++)
				c[i] = a[i] + b[i];
		});
	}).wait();       // End of the queue commands we waint on the event reported.
}

void VectorAdd2(sycl::queue &Q, int *a, int *b,
	int *c, int N, int iter, int num_groups) {

	size_t wg_size = 256;
	int iters = N/num_groups/wg_size;

	Q.submit([&](handler &h) {
		range global = range<1>(num_groups * wg_size);
		range local = range<1>(wg_size);
		h.parallel_for(nd_range<1>(global, local),[=](nd_item<1> index)
			 [[intel::reqd_sub_group_size(32)]] {
			size_t grp_id = index.get_group()[0];
			size_t loc_id = index.get_local_id();
			size_t start = (grp_id*wg_size*iters+loc_id);
			size_t end = start + iters*wg_size;

			for (int it = 0; it < iter; it++)
				for (size_t i = start; i < end; i+=wg_size) {
					c[i] = a[i] + b[i];
				}
		}); 	// End of the kernel function
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

	sycl::queue Q(sycl::gpu_selector_v);

	std::cout << "Running on "
		<< Q.get_device().get_info<sycl::info::device::name>()
		<< std::endl;


	int *a = malloc_shared<int>(N, Q);
	int *b = malloc_shared<int>(N, Q);
	int *c = malloc_shared<int>(N, Q);
	int *c_gt = static_cast<int *>(malloc(N * sizeof(int)));

	// Parallel for
	for(int i=0; i<N; i++){
		a[i] = i;   // Init a
		b[i] = i*i; // Init b
		c_gt[i] = a[i] + b[i];
	}

	dummy(Q, N);

	/* benchmarking */
	auto start = std::chrono::steady_clock::now();
	VectorAdd1(Q, a, b, c, 100, N);
	auto end = std::chrono::steady_clock::now();
	printf("Time VectorAdd1=%ld usecs\n", (end - start).count());

	for (int ng=1; ng<64; ng*=2) {
		start = std::chrono::steady_clock::now();
		VectorAdd2(Q, a, b, c, N, 100, ng);
		end = std::chrono::steady_clock::now();
		printf("Time VectorAdd2=%ld usecs (num work_groups=%d)\n", (end - start).count(), ng);
		if (!test(c_gt, c, N))
			printf("Error in VectorAdd2\n");
	}

	free(a, Q);
	free(b, Q);
	free(c, Q);
	free(c_gt);

	return 0;
}
