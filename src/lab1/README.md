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
cpu MHz : 2100.000
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

