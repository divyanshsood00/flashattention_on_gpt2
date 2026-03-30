mkdir -p minitorch/cuda_kernels
# nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
nvcc -arch=sm_86 -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
