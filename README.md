# Complex polynomials arithmetic

Maven project coded in Java. Contains two modules.  
- **cplx-cuda-binders** is a set of utilities used to interact with CUDA from JAVA. It's mostly boilerplate code I used in other projects and not really relevant.
- **cplx-polynomial-routines** contains the code and kernels for polynomial operations.

Production code is in src/main/java while test code is in src/test/java.

- **ComplexPolynomialOperations.java** is the entry point to all routines. The methods take CPU objects ComplexPolynomial as inputs and outputs. It's the API layer.
- **GPUAdd.java** Implementation of Polynomial Add on GPU.
- **GPUMultiply.java** Implementation of Polynomial Multiplication on GPU.
- **GPUMatrixAdd.java** Implementation of Matrix Polynomial Add on GPU.
- **GPUMatrixMultiply.java** Implementation of Matrix Polynomial Multiplication on GPU.

The project is tested on Windows using a 3080Ti (Should be working on Linux as well but untested)

# How to build and run
## Requirements
- JDK 18 (https://www.oracle.com/java/technologies/javase/jdk18-archive-downloads.html)  
- A GPU capable of running CUDA 11.8+ and up to date nVidia drivers. The code is build dynamically using NVRTC so CUDA toolkit should not be necessary to run. I would be necessary to profile, though.  
- Maven (https://maven.apache.org/download.cgi)  

## Build
- Ensure Java is in your path (java --version)  
- Ensure Maven is in your path (mvn -v)  
- Checkout the code and navigate to the folder.  
- Run ```mvn clean install```. This downloads all dependencies and runs all tests. 
- A prebuilt jarfile is provided as a release artifact.

## Execute 
- The benchmarking binary should be created at cplx-polynomial-routines/target/cplx-jar-with-dependencies.jar  
- Display it's help to see the available commands :  
```java -jar cplx-polynomial-routines/target/cplx-jar-with-dependencies.jar```
- Run one of the benchmarks :  
```java -Xmx4g -jar cplx-polynomial-routines/target/cplx-jar-with-dependencies.jar MUL 22```
- Run the full benchmark suite :  
```java -Xmx32g -jar cplx-polynomial-routines/target/cplx-jar-with-dependencies.jar BENCHMARK```

# How to profile
- Run NVIDIA Nsight compute as administrator
- Create a new project
- Application executable should be java.exe (something like C:/Program Files/Java/jdk-18.0.2.1/bin/java.exe)
- Application arguments : ```-Xmx16g path/to/cplx-jar-with-dependencies.jar MUL 22```

# Sample benchmark results

Benchmarking polynomial Addition...  
Creating two polynomials of size 1024        Adding... Done in 0 ms  
Creating two polynomials of size 2048        Adding... Done in 0 ms  
Creating two polynomials of size 4096        Adding... Done in 0 ms  
Creating two polynomials of size 8192        Adding... Done in 0 ms  
Creating two polynomials of size 16384       Adding... Done in 0 ms  
Creating two polynomials of size 32768       Adding... Done in 0 ms  
Creating two polynomials of size 65536       Adding... Done in 1 ms  
Creating two polynomials of size 131072      Adding... Done in 1 ms  
Creating two polynomials of size 262144      Adding... Done in 2 ms  
Creating two polynomials of size 524288      Adding... Done in 2 ms  
Creating two polynomials of size 1048576     Adding... Done in 3 ms  
Creating two polynomials of size 2097152     Adding... Done in 6 ms  
Creating two polynomials of size 4194304     Adding... Done in 12 ms  
Creating two polynomials of size 8388608     Adding... Done in 24 ms  
Creating two polynomials of size 16777216    Adding... Done in 48 ms  
Creating two polynomials of size 33554432    Adding... Done in 95 ms  
  
Benchmarking polynomial multiplication...  
Creating two polynomials of size 1024        Multiplying... Done in 1 ms  
Creating two polynomials of size 2048        Multiplying... Done in 1 ms  
Creating two polynomials of size 4096        Multiplying... Done in 1 ms  
Creating two polynomials of size 8192        Multiplying... Done in 2 ms  
Creating two polynomials of size 16384       Multiplying... Done in 1 ms  
Creating two polynomials of size 32768       Multiplying... Done in 2 ms  
Creating two polynomials of size 65536       Multiplying... Done in 2 ms  
Creating two polynomials of size 131072      Multiplying... Done in 2 ms  
Creating two polynomials of size 262144      Multiplying... Done in 3 ms  
Creating two polynomials of size 524288      Multiplying... Done in 5 ms  
Creating two polynomials of size 1048576     Multiplying... Done in 10 ms  
Creating two polynomials of size 2097152     Multiplying... Done in 17 ms  
Creating two polynomials of size 4194304     Multiplying... Done in 34 ms  
Creating two polynomials of size 8388608     Multiplying... Done in 68 ms  
Creating two polynomials of size 16777216    Multiplying... Done in 139 ms  
Creating two polynomials of size 33554432    Multiplying... Done in 276 ms  
  
Benchmarking matrix addition...  
Creating 2 * [64][64]     size 256  polynomials     Adding... Done in 22 ms  
Creating 2 * [128][128]   size 256  polynomials     Adding... Done in 49 ms  
Creating 2 * [256][256]   size 256  polynomials     Adding... Done in 137 ms  
Creating 2 * [512][512]   size 256  polynomials     Adding... Done in 501 ms  
Creating 2 * [1024][1024] size 256  polynomials     Adding... Done in 1992 ms  
Creating 2 * [2048][2048] size 256  polynomials     Adding... Done in 5819 ms  

Benchmarking matrix multiplication...  
Creating 2 * [16][16]     size 1024 polynomials     Multiplying... Done in 11 ms  
Creating 2 * [32][32]     size 1024 polynomials     Multiplying... Done in 38 ms  
Creating 2 * [64][64]     size 1024 polynomials     Multiplying... Done in 153 ms  
Creating 2 * [128][128]   size 1024 polynomials     Multiplying... Done in 280 ms  
Creating 2 * [256][256]   size 1024 polynomials     Multiplying... Done in 1027 ms  
Creating 2 * [512][512]   size 1024 polynomials     Multiplying... Done in 5479 ms  
Creating 2 * [1024][1024] size 1024 polynomials     Multiplying... Done in 54162 ms  

# Future work / Optimizations

- [ ] CPU Objects representation is too different from GPU representation, this makes a few kernels too complex.
- [ ] Multi-GPU should be easy to implement on matrix multiplication since we already have slicing in place.
- Investigate hotspots :
	- [ ] FFT Preparation kernel can likely be sped up on simple multiplication. It's coalescence is not optimal.
	- [ ] Matrix multiplication of the FFTs is the worst offender in matrix multiplication, we can probably optimize it.
	- [ ] Thread copy and execution of simple multiplication with one buffer per polynomial instead of a shared one.
- [ ] Up to the first 5 rounds (because 2^5 = 32 = wrap size) of the FFT can probably be run in a single kernel using shfl_up_sync(). This would save a lot of global reads / writes.
- [ ] Mutualize some of the CPU code.