package eth.bram.cplx.kernels.simple;

import static java.nio.ByteOrder.LITTLE_ENDIAN;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import eth.bram.cplx.ComplexPolynomial;
import eth.bram.cplx.GPUBuffer;
import eth.bram.cplx.GPUContext;
import eth.bram.cplx.GPUKernel;
import eth.bram.cplx.GPUProgram;
import eth.bram.cplx.kernels.CudaSource;

/**
 * GPU implementation of the multiplication of two complex polynomials.
 * 
 * This version relies on FFT and performs the following :
 * 	- Computes the FFT of 2 polynomials.
 *  - Computes the multiplication a.b
 *  - Reverses the FFT of the product.
 * 
 * The GPU based FFT requires a pre-step to align data in the correct way. 
 * This step is performed once before FFT and once before the reverse FFT.
 * 
 * We work with polynomials of double the size provided (padded with zeros) since the multiplication
 * would lead to a polynomial of size 2n.
 * 
 * This version is faster than the naive O(n^2) version because FFT is O(nlogn) and multiplication is O(n).
 */
public class GPUMultiply {

	/**
	 * GPU context wrapper
	 */
	private final GPUContext ctx;

	/**
	 * FFT preparation kernel (Polynomial version, expands the polynomial from 2^n to 2^2n)
	 */
	private final GPUKernel fftPreparePolynomialKernel;
	
	/**
	 * FFT kernel
	 */
	private final GPUKernel fftKernel;

	/**
	 * FFT First kernel
	 */
	private final GPUKernel fftFirstKernel;
	
	/**
	 * Multiply kernel
	 */
	private final GPUKernel multiplyKernel;
	
	/**
	 * Finalization of the multiplication kernel
	 */
	private final GPUKernel finalizeMultiplicationKernel;
	
	/**
	 * Polynomials data (input)
	 */
	private final GPUBuffer inPolynomials;
	
	/**
	 * GPU buffer holding in place FFT data
	 */
	private final GPUBuffer inOutFFT;
	
	/**
	 * Result of the multiplication
	 */
	private final GPUBuffer outMul;
	
	/**
	 * Polynomial size
	 */
	private final int polynomialSize;
	
	/**
	 * Creates context and buffers on the given device.
	 * 
	 * @param deviceId index of the cuda device to use.
	 * @param polynomialSize size of the polynomials we work with.
	 */
	public GPUMultiply(int deviceId, int polynomialSize) {
		
		// Check polynomial size is a power of two
		if (Integer.bitCount(polynomialSize) != 1) {
			throw new IllegalArgumentException("Polynomial size must be a power of 2");
		}
		
		// Double the size of the polynomials - we need to pad inputs to 2^n+1 to keep the full multiplication data when doing FFT
		polynomialSize *= 2;
		
		this.polynomialSize = polynomialSize;
		this.ctx = GPUContext.create(deviceId);
		final GPUProgram program = ctx.createProgram(Arrays.asList(CudaSource.getSource("kernels.cu")));
		
		// Polynomials are copied in their 2^n form
		this.inPolynomials = ctx.createBuffer(ByteBuffer.allocate(polynomialSize * 2 * Float.BYTES).order(LITTLE_ENDIAN));
		
		// During preparation for the FFT we will expand the 2^n in 2^(2n) form for multiplication, hence the doubling in size
		this.inOutFFT = ctx.createBuffer(ByteBuffer.allocate(this.polynomialSize * 2 * Float.BYTES).order(LITTLE_ENDIAN));
		
		// multiplication output only requires 2^n form
		this.outMul = ctx.createBuffer(ByteBuffer.allocate(this.polynomialSize * Float.BYTES).order(LITTLE_ENDIAN));
		
		// FFT Prepare polynomial kernel and parameters
		this.fftPreparePolynomialKernel = program.buildKernel("fftPreparePolynomial");
		fftPreparePolynomialKernel.setArg(0, inPolynomials.pointer());
		fftPreparePolynomialKernel.setArg(1, inOutFFT.pointer());
		fftPreparePolynomialKernel.setArg(2, rounds());
		
		// FFT Kernel and parameters
		this.fftKernel = program.buildKernel("fft");
		fftKernel.setArg(0, inOutFFT.pointer());
		
		// multiplication kernel
		this.multiplyKernel = program.buildKernel("multiply");
		multiplyKernel.setArg(0, inOutFFT.pointer());
		multiplyKernel.setArg(1, outMul.pointer());
		multiplyKernel.setArg(2, polynomialSize);
		
		// FFT first round
		this.fftFirstKernel = program.buildKernel("fftFirst");
		fftFirstKernel.setArg(0, outMul.pointer());
		fftFirstKernel.setArg(1, inOutFFT.pointer());
		fftFirstKernel.setArg(2, rounds());
		
		// Finalize multiplication
		this.finalizeMultiplicationKernel = program.buildKernel("finalizeMultiplication");
		finalizeMultiplicationKernel.setArg(0, inOutFFT.pointer());
		finalizeMultiplicationKernel.setArg(1, inPolynomials.pointer());
		finalizeMultiplicationKernel.setArg(2, polynomialSize);
	}

	/**
	 * Performs FFT based multiplication of polynomials.
	 * 
	 * @return a A * B.
	 */
	public ComplexPolynomial mul(ComplexPolynomial a, ComplexPolynomial b) {

		// Check buffer sizes are correct
		checkSize(a.re());
		checkSize(a.im());
		checkSize(b.re());
		checkSize(b.im());
		
		// Send polynomials to GPU
		inPolynomials.buffer().put(a.re()).put(a.im()).put(b.re()).put(b.im()).rewind();
		inPolynomials.write();
		
		// Prepare data for FFT for the 2 polynomials
		fftPreparePolynomialKernel.run(polynomialSize * 2);
		
		// Run FFT kernels on both polynomials, log2(polynomialSize) rounds.
		final int rounds = rounds();
		for (int i = 1 ; i <= rounds ; ++i) {
			fftKernel.setArg(1, 1 << i);
			fftKernel.run(polynomialSize);
		}
		
		// Perform multiplication
		multiplyKernel.run(polynomialSize);
		
		// Reverse FFT on the result of the multiplication
		// The first kernel is fetching data in a different place.
		fftFirstKernel.run(polynomialSize / 2);
		for (int i = 2 ; i <= rounds ; ++i) {
			fftKernel.setArg(1, 1 << i);
			fftKernel.run(polynomialSize / 2);
		}
		
		// Finalize multiplication and compress result.
		// This kernel brings us back from 2n polynomials to n.
		// Each thread copies 2 values from the input to the output, hence the (polynomialSize / 2) threads.
		finalizeMultiplicationKernel.run(polynomialSize / 2);

		// Read and return result
		inPolynomials.read();
		return new ComplexPolynomial(inPolynomials.buffer().slice(0, a.re().capacity()).order(ByteOrder.LITTLE_ENDIAN), 
							 		 inPolynomials.buffer().slice(a.re().capacity(), a.im().capacity()).order(ByteOrder.LITTLE_ENDIAN));
	}

	/**
	 * Checks the size of the buffer matches the polynomial size provided.
	 */
	private void checkSize(ByteBuffer b) {
		if(b.capacity() != polynomialSize * Float.BYTES / 2) {
			throw new IllegalArgumentException("Provided buffer does not match the kernel polynomial size");
		}
	}

	/**
	 * Power of two of the polynomial size.
	 */
	private int rounds() {
		return (int)(Math.log(polynomialSize) / Math.log(2));
	}

	/**
	 * Release all created objects in this context : buffers, kernels, streams...
	 */
	public void release() {
		ctx.release();
	}
}
