# Códigos del laboratorio 1: "Programación de GPUs con CUDA"
## Objetivos
* Familiarizarse con la arquitectura CUDA
* Evaluar las mejoras/speedup de GPU vs CPU

# Arquitectura GPUs
## Arquitectura G80/G90/GT200
* Nuevo modelo arquitectura: Arquitectura unificada de shaders 
* Solo hay un tipo de unidad programable para pixels/vertex 
    * Ejecuta todo los tipos de shaders
* Estructura realimentada

![Imagen](figures/discrete_shaders_archicture.png)
![Imagen](figures/unified_shaders_architecture.png)

## Arquitectura G80/G90/GT200
* 8 clusters
* Cada cluster tiene un planificador compartida por los 16 SP (ALUs), y unidades de textura

### GPUs vistas como procesador de hilos
![Imagen](figures/g80arch.png)

# CUDA 
## Entorno desarrollo
* Instalado CUDA 11.7
    * Ubuntu 22.04 por defecto instalado gcc-10

## Compiladores
* *nvcc* para compilar los ficheros fuente .cu (kernels)
* *g++/gcc* para compilar los ficheros fuente .cpp/.c y enlazar

### nvidia-smi

``` bash
usuario_local@profess11:~$ nvidia-smi
Fri Feb 17 16:02:47 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.86.01    Driver Version: 515.86.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 25%   41C    P8    21W / 160W |    463MiB /  6144MiB |      3%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```


### nvcc
\begin{console}{Terminal \#1}
\tiny
\begin{verbatim}
usuario_local@profess11:~$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Wed_Jul_22_19:09:09_PDT_2020
Cuda compilation tools, release 11.0, V11.0.221
Build cuda_11.0_bu.TC445_37.28845127_0
\end{verbatim}
\end{console} 


* Procesador Intel i7-8700 con 6 cores + SMT

### cpu_info
``` bash
usuario_local@profess11:~$ more /proc/cpuinfo
processor : 0
vendor_id : GenuineIntel
cpu family : 6
model : 151
model name : 12th Gen Intel(R) Core(TM) i7-12700
stepping : 2
microcode : 0x25
cpu MHz : 2100.000![Imagen](figures/discrete_shaders_archicture.png)
cache size : 25600 KB
physical id : 0
siblings : 20
core id : 0
cpu cores : 12
apicid : 0
initial apicid : 0
fpu : yes
fpu_exception : yes
cpuid level : 32
wp : yes
flags : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe sys call nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclm ulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave  avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept  vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb intel_pt sha_ni xsaveopt xsavec xgetbv1 xsa ves split_lock_detect avx_vnni dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp hwp_pkg_req umip pku ospke waitpkg gfni vaes vpc lmulqdq tme rdpid movdiri movdir64b fsrm md_clear serialize pconfig arch_lbr flush_l1d arch_capabilities vmx flags : vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs ept_mode_based_exec tsc_scaling usr_wait_pause
bugs : spectre_v1 spectre_v2 spec_store_bypass swapgs
bogomips : 4224.00
clflush size : 64
```

# Ejemplos para trabajar
* En este repositorio se encuentran los códigos para que el alumnado conozca las principales caracteristicas de programación de GPUs con el modelo de CUDA
* Antes de nada vamos a conocer las características de la GPU con que trabajaremos en el laboratorio mediante el [código **device_info**](device_info/device_info.cu)
    * Para compilar vamos a utilizar el compilador [NVIDIA CUDA Compiler Driver NVCC **nvcc**](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
    * En el fichero [README](device_info/README) muestra la información para compilar el código anterior: ```nvcc device_info.cu``` que generará el ejecutable **a.out**


## Suma matrices
* Partiendo el código siguiente que muestra el esquema de codificación de la suma de vectores en una CPU

``` c
// Compute vector sum C = A+B
void vecAdd(float* A, float* B, float* C,
   int n)
{
   for (i = 0, i < n, i++)
      C[i] = A[i] + B[i];
}

int main()
{
   // Memory allocation for A_h, B_h, C_h
   // I/O to read A_h and B_h, N elements
   ...
   vecAdd(A_h, B_h, C_h, N);
}
```

* Podemos adaptar esa misma idea para a un código CUDA con la construcción del kernel **vecAddkernel**

```c
// Compute vector sum C = A+B
__global__
void vecAddkernel(float* A_d, float* B_d, float* C_d, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i<n) C_d[i] = A_d[i] + B_d[i];
}

int main()
{
   float* A_d, B_d, C_d;
   int size = n* sizeof(float); 

   // Get device memory for A, B, C
   // copy A and B to device memory
   cudaMalloc((void **) &A_d, size);
   cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
   cudaMalloc((void **) &B_d, size);
   cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
   cudaMalloc((void **) &C_d, size);
    
   // Kernel execution in device
   // (vector add in device)
   vecAddkernel<<<nBlocks, nThread_per_Blocks>>>(A_d, B_d, C_d, n);

   // copy C from device memory
   cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
   // free A, B, C
   cudaFree(A_d); cudaFree(B_d); cudaFree (C_d);
}
```

* En el [código de suma de matrices](matrix_add/main.cu) implementa un esqueleto para la suma de matrices. La función **addMatrix** que se presenta a continuación realiza la suma de matrices **b** y **c** en la CPU. Para compilar dicho código se empleará el compilador **nvcc** como en el ejemplo anterior

```c
void addMatrix(float *a, float *b, float *c, int N)
{
	int i, j, idx;
	for (i=0; i<N; i++)
		for(j=0; j<N;j++){
			idx = i*N+j;
			a[idx]=b[idx]+c[idx];
		}
}
```

* En este ejemplo se pide que el alumnado complete el [código](matrix_add/main.cu), prestando especial atención a:
    1. La reserva de memoria con **cudaMalloc(...)** donde se ha de especificar el tamaño a reservar
    2. La copia de memoria desde el *host* al *device* con las instrucciones **cudaMemcpy(...)**
    3. Indicar el número de bloques e hilos (**dim3 dimBlock(...,...);** y **dim3 dimGrid(...,...)**) de bloques para el kernel ```addMatrixGPU<<<dimGrid,dimBlock>>>(a_GPU, b_GPU, c_GPU, N);```
    4. Rellenar el kernel **addMatrixGPU** que aparece a continuación
    5. Liberar la memoria de la GPU con las instrucciones **cudaFree(...);**

```c
__global__ void addMatrixGPU(float *a, float *b, float *c, int N )
{
	....
}
```

## Multiplicación de matrices
* En este ejemplo vamos a trabajar la multiplicación de matrices descrita como $C_{NM}=A_{NK}*B_{KM}$
    * El código se encuentra en el directorio [**matrix_mult**](matrix_mult/)
```c
   for (i = 0; i < N; i++) {
      for (j = 0; j < M; j++) {
         for (k = 0; k < K; k++) {
            C[i][j] += A[i][k]*B[k][j];
        }
      }
   }
```

    * El fichero [**main.c**](matrix_mult/main.c) lanza la ejecución en la GPU en **Mul(A, B, hA, wA, wB, C);** que se encuentra en el fichero [**matrix_mult.cu**](matrix_mult/matrix_mul.cu)
    * El kernel hay que rellenarlo puesto que está vacío

```c
__global__ void Muld(float* A, float* B, int wA, int wB, float* C)
{
	//To Do
}
```

![Imagen](figures/cuda-matrix-multiplication-with-shared-memory.png)

### To Do
* Rellenar el kernel **Muld** del fichero [**matrix_mult.cu**](matrix_mult/matrix_mul.cu)
    * Se puede comenzar con una implementación 1D
    * Para continuar con la implementación 2D que es la que aparece en el fichero [**matrix_mult.cu**](matrix_mult/matrix_mul.cu)

### Implementación con memoria compartida
* Muchos threads
    * Un Thread por cada Elemento de C[i][j]
    * Cada Thread necesita Fila A[i][:] y Columna B[:][j]
* Podemos Usar Memoria Compartida
    * Cada Block Recorre Filas A[-][:] y Columnas B[:][-]
    * Hay Localidad a nivel de CUDA Block!!



![Imagen](figures/cuda-matrix-multiplication.png)

### Implementación mediante librerías
* Haremos uso de la librerías CUBLAS
    * BLAS (Basic Linear Algebra Subroutine)
    * CUBLAS = CUDA + BLAS
    * Operación GEMM:
        * C = $\alpha$ op ( A ) op ( B ) + $\beta$ C
        * $\alpha$ y $\beta$ son escalares
        * $A_{m \times k}$, $B_{k \times n}$ y $C_{n \times n}$ en (column-major)

```
cublasStatus_t cublasSgemm(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float           *alpha,
    const float           *A, int lda,
    const float           *B, int ldb,
    const float           *beta,
    float                 *C, int ldc)
```

* mulMat $\rightleftharpoons$ cublasSgemm
    * Sustitución de invocación a **Muld** por cublasSgemm
    * A tener en cuenta
        * column-major vs. row-major
        * $A*B = (B^T*A^T)^T$

![Imagen](figures/rowcolumnarrays.jpg)

### ToDo
* Comparativa de tiempos de ejecución (GPU-CPU) para la multiplicación de matrices
    * A[4096][1024]*B[1024][2048]
    * A[1024][1024] *B[1024][1024]
    * A[8192][512] *B[512][4096]


