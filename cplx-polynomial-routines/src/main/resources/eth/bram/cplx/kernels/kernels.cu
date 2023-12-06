#define PI 3.14159265358979323846f
#define GLOBAL_IDX (blockIdx.x * blockDim.x + threadIdx.x)
#define GLOBAL_SIZE (blockDim.x * gridDim.x)

/**
* Adds two complex values
*/
__device__  inline float2 add(float2 a, float2 b) {
  float2 c; 
  c.x = a.x + b.x; 
  c.y = a.y + b.y;
  return c;
}

/**
* Substracts a complex value
*/
__device__  inline float2 sub(float2 a, float2 b) {
  float2 c; 
  c.x = a.x - b.x; 
  c.y = a.y - b.y;
  return c;
}

/**
* Multiplies two complex values
*/
__device__  inline float2 mul(float2 a, float2 b) {
  float2 c; 
  c.x = a.x * b.x - a.y * b.y; 
  c.y = a.x * b.y + a.y * b.x; 
  return c;
}

/**
* Conjugates a complex value
*/
__device__  inline float2 conjugate(float2 a) {
  float2 c; 
  c.x = a.x; 
  c.y = -a.y; 
  return c;
}

/**
* Scales a complex value
*/
__device__  inline float2 scale(float2 a, float alpha) {
  float2 c; 
  c.x = a.x * alpha; 
  c.y = a.y * alpha; 
  return c;
}

/**
* Polynomial addition kernel. 
* Use float2 to optimize ILP
*/
extern "C" __global__ void add(float2 * A, float2 * B, float2 * res)
{
    float2 a = A[GLOBAL_IDX];
    float2 b = B[GLOBAL_IDX];
    float2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
	res[GLOBAL_IDX] = c;
}

/**
 * First FFT step feching data in the correct location.
 * Example layout for an 0 1 2 3 4 5 6 7 polynomial : 0 4 2 6 1 5 3 7
 * 
 * @param in data to order for FFT
 * @param out data ordered for FFT
 * @param rounds number of fft rounds which will be run.
 * @param size size of the input object.
 */
extern "C" __global__ void fftFirstStep(float2 * in, float2 * out, int rounds, int itemSize) {

	// Offset buffer according to which slice we will be working on
	const int idx = GLOBAL_IDX / (itemSize / 2);
	const int idxInSlice = GLOBAL_IDX % (itemSize / 2);
	in += idx * itemSize;
	out += idx * itemSize;

	// k = 0
    float2 w;
    w.x = 1; // cos(0)
    w.y = 0; // sin(0)
    
    // Compute and propagate for next iteration
	float2 q = in[__brev(idxInSlice * 2) >> (32 - rounds)];
	float2 r = in[__brev(idxInSlice * 2 + 1) >> (32 - rounds)];
	w = mul(w, r);
	
	__shared__ float2 localFFT [64];
	localFFT[idxInSlice * 2] = add(q, w);
	localFFT[idxInSlice * 2 + 1] = sub(q, w);
	
    int size = 2;
    while(size < 4) {
    	
    	// Bump size
    	size *= 2;
    	
    	// Offset
    	int offset = ((idxInSlice * 2) / size) * size;
    	
    	// Item we are handling in the subset
    	int k = idxInSlice % (size / 2);
    	
    	// Compute kth root of unity
        float kth = (-2.0f * PI * k) / size;
        float2 w;
        w.x = cos(kth);
        w.y = sin(kth);
        
        __syncthreads();
    	q = localFFT[offset + k];
    	r = localFFT[offset + k + size / 2];
    	
    	w = mul(w, r);
    	localFFT[offset + k] = add(q, w);
    	localFFT[offset + k + size / 2] = sub(q, w);
    }
	
    // Store last result
    __syncthreads();
    out[idxInSlice * 2] = localFFT[idxInSlice * 2];
    out[idxInSlice * 2 + 1] = localFFT[idxInSlice * 2 + 1];
}

/**
* FFT kernel using Cooley–Tukey method. This kernel is run log2(n) times.
* @param inOut : data buffer, contains the current state of the FFT computation.
* @param size : size of the subset we are processing.
*/
extern "C" __global__ void fft(float2 * inOut, int size) {
	    
	// Position buffer on the correct subset
	inOut += ((GLOBAL_IDX * 2) / size) * size;

	// Item we are handling in the subset
	int k = GLOBAL_IDX % (size / 2);
	
	// Compute kth root of unity
    float2 w;
    float kth = (-2.0f * PI * k) / size;
    w.x = cos(kth);
    w.y = sin(kth);
    
    // Compute and propagate for next iteration
	float2 q = inOut[k];
	float2 r = inOut[k + size / 2];
	w = mul(w, r);
    inOut[k] = add(q, w);
    inOut[k + size / 2] = sub(q, w);
}

/**
* Complex point to point mulitiplication (Equivalent of polynomial multiply in the FFT domain)
* For efficiency reasons, this kernel also conjugates the return values to prepare for following FFT Inversion.
* 
* @param A polynomial A in FFT form.
* @param B polynomial B in FFT form.
*/
extern "C" __global__ void multiply(float2 * A, float2 * B) {
	A[GLOBAL_IDX] = conjugate(mul(A[GLOBAL_IDX], B[GLOBAL_IDX]));
}

/**
* Complex point to point mulitiplication for matrices (Equivalent of polynomial multiply in the FFT domain)
* For efficiency reasons, this kernel also conjugates the return values to prepare for following FFT Inversion.
* 
* @param A set of polynomials in FFT form.
* @param B set of polynomials in FFT form.
* @param out buffer to write results to.
* @param polynomialSize size of the polynomials
* @param columnCount number of columns we are processing
* @param rowCount number of rows we are processing
* @param width width of a matrix A line (or matrix B column)
*/
#define MAX_ROWS_REG 8
#define MAX_ROWS_SHARED 8
#define UNROLL_SIZE 4
#define LOCAL_SIZE_MATRIX_MULTIPLY <LOCAL_SIZE_MATRIX_MULTIPLY>
extern "C" __global__ void multiplyMatrix(float2 * A, float2 * B, float2 * out, int polynomialSize, int columnCount, int rowCount, int width) {
	
	// Offset computation for buffers
	const int polynomialId = GLOBAL_IDX / polynomialSize;
	const int polynomialElement = GLOBAL_IDX % polynomialSize;
	const int col =  polynomialId % columnCount;
	
	// Init value
	float2 zero;
	zero.x = 0.0;
	zero.y = 0.0;

	// Store part of the data in registers
	float2 sumsReg [MAX_ROWS_REG];
	#pragma unroll MAX_ROWS_REG
	for (int line = 0 ; line < MAX_ROWS_REG ; ++line) {
		sumsReg[line] = zero;
	}
	
	// Store part of the data in local memory
	__shared__ float2 sumsShared [MAX_ROWS_SHARED * LOCAL_SIZE_MATRIX_MULTIPLY];
	#pragma unroll MAX_ROWS_SHARED
	for (int line = 0 ; line < MAX_ROWS_SHARED ; ++line) {
		sumsShared[BLOCK_IDX + BLOCK_SIZE * line] = zero;
	}
	
	// Sum the point to point multiplication
	for (int i = 0 ; i < width ; ++i) {
		const float2 b = B[col * width * polynomialSize + i * polynomialSize + polynomialElement];
		
		int line = 0;
		#pragma unroll UNROLL_SIZE
		for ( ; line < rowCount && line < 8 ; ++line) {
			float2 a = A[line * width * polynomialSize + i * polynomialSize + polynomialElement];
			float2 data = sumsReg[line];
			data.x += a.x * b.x - a.y * b.y; 
			data.y += a.x * b.y + a.y * b.x; 
			sumsReg[line] = data;
		}
		
		#pragma unroll UNROLL_SIZE
		for ( ; line < rowCount ; ++line) {
			float2 a = A[line * width * polynomialSize + i * polynomialSize + polynomialElement];
			float2 data = sumsShared[BLOCK_IDX + BLOCK_SIZE * (line - MAX_ROWS_REG)];
			data.x += a.x * b.x - a.y * b.y; 
			data.y += a.x * b.y + a.y * b.x; 
			sumsShared[BLOCK_IDX + BLOCK_SIZE * (line - MAX_ROWS_REG)] = data;
		}
	}
	
	// Store in global
	int line = 0;
	#pragma unroll UNROLL_SIZE
	for ( ; line < rowCount && line < 8 ; ++line) {
		out[line * polynomialSize * columnCount + col * polynomialSize + polynomialElement] = conjugate(sumsReg[line]);
	}
	
	#pragma unroll UNROLL_SIZE
	for ( ; line < rowCount ; ++line) {
		out[line * polynomialSize * columnCount + col * polynomialSize + polynomialElement] = conjugate(sumsShared[BLOCK_IDX + BLOCK_SIZE * (line - MAX_ROWS_REG)]);
	}
}

/**
 * Finalize the multiplication : 
 *  - Conjugate and scale the FFT input
 * @param in  input complex polynomial to scale (size 2n)
 * @param out buffer to write multiplication result to (size n)
 */
extern "C" __global__ void finalizeMultiplication(float2 * in, float2 * out, int polynomialSize) {
	const int polynomialIdx = GLOBAL_IDX / polynomialSize;
	const int idxInPolynomial = GLOBAL_IDX % polynomialSize;
	in += polynomialIdx * polynomialSize * 2;
	out += polynomialIdx * polynomialSize;
	out[idxInPolynomial] = scale(conjugate(in[idxInPolynomial]), 1.0f / (polynomialSize * 2));
}