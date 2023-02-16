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
* Instalado CUDA 11.1
    * Ubuntu 22.04 por defecto instalado gcc-7.5

## Compiladores
* *nvcc* para compilar los ficheros fuente .cu (kernels)
* *g++/gcc* para compilar los ficheros fuente .cpp/.c y enlazar

### nvidia-smi

``` bash
usuario_local@profess11:~$ nvidia-smi
Fri Feb 14 14:45:31 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 2060    Off  | 00000000:01:00.0  On |                  N/A |
| 26%   37C    P8    13W / 160W |    677MiB /  5931MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      5294      G   /usr/bin/gnome-shell                          97MiB |
|    0      6955      G   /usr/lib/xorg/Xorg                           229MiB |
|    0     11164      G   ...quest-channel-token=7661568818610171974   348MiB |
+-----------------------------------------------------------------------------+
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
processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 158
model name	: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
stepping	: 10
microcode	: 0xca
cpu MHz		: 800.073
cache size	: 12288 KB
physical id	: 0
siblings	: 12
core id		: 0
cpu cores	: 6
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single pti ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp md_clear flush_l1d
```

