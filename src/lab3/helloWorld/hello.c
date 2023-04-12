

#include <stdio.h>

#ifdef _OPENACC
#include <openacc.h>
#endif

#define N 1000

int main() {

   int a[N];
   int b[N];

   #ifdef _OPENACC
     acc_init(acc_device_not_host);
     printf(" Compiling with OpenACC support \n");
   #endif 

    printf(" Hello World! \n ");

   // Compute on the host
   for (int i = 0; i < N; i++) {
      a[i] = i;
    }
   // Compute on the GPU if OpenACC support - host if not
   #pragma acc kernels copy(b[0:N])
   for (int i = 0; i < N; i++) {
      b[i] = i;
    }
   for (int i = 0; i < N; i++) {
      if (a[i] != b[i])  {
		printf("Something went wrong\n");
		return 1;
      }
   }
   #ifdef _OPENACC
     acc_shutdown(acc_device_not_host);
   #endif 

   return 0;

}
