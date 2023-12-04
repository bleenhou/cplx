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
 * Populates the values in a nice order for the reverse FFT computation.
 * Example layout for an 0 1 2 3 4 5 6 7 polynomial : 0 4 2 6 1 5 3 7
 * 
 * @param  in data to order for FFT
 * @param  out data ordered for FFT
 * @params rounds number of fft rounds which will be run.
 */
extern "C" __global__ void fftPrepareComplex(float2 * in, float2 * out, int rounds, int size) {

	// Offset buffer according to which slice we will be working on
	const int idx = GLOBAL_IDX / size;
	const int idxInSlice = GLOBAL_IDX % size;
	in += idx * size;
	out += idx * size;
	
	// Order for FFT
	const int destIdx = __brev(idxInSlice) >> (32 - rounds);
	out[destIdx] = in[idxInSlice];
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
* @param in set of polynomials in FFT form.
* @param out buffer to write results to.
* @param polynomialSize size of the polynomials.
*/
extern "C" __global__ void multiply(float2 * in, float2 * out, int polynomialSize) {
	const float2 a = in[GLOBAL_IDX];
	const float2 b = in[polynomialSize + GLOBAL_IDX];
	out[GLOBAL_IDX] = conjugate(mul(a, b));
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
#define MAX_ROWS 8
extern "C" __global__ void multiplyMatrix(float2 * A, float2 * B, float2 * out, int polynomialSize, int columnCount, int rowCount, int width) {
	
	// Offset computation for buffers
	const int polynomialId = GLOBAL_IDX / polynomialSize;
	const int polynomialElement = GLOBAL_IDX % polynomialSize;

	const int col =  polynomialId % columnCount;
	B += col * width * polynomialSize;
	
	float2 sums [MAX_ROWS];
	for (int i = 0 ; i < MAX_ROWS ; ++i) {
		sums[i].x = 0.0;
		sums[i].y = 0.0;
	}
	
	// Sum the point to point multiplication
	for (int i = 0 ; i < width ; ++i) {
		const float2 b = B[i * polynomialSize + polynomialElement];
		for (int line = 0 ; line < rowCount ; ++line) {
			float2 a = A[line * width * polynomialSize + i * polynomialSize + polynomialElement];
			sums[line].x += a.x * b.x - a.y * b.y; 
			sums[line].y += a.x * b.y + a.y * b.x; 
		}
	}
	
	// Store in global
	for (int line = 0 ; line < rowCount ; ++line) {
		out[line * polynomialSize * columnCount + col * polynomialSize + polynomialElement] = conjugate(sums[line]);
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