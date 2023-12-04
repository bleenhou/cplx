package eth.bram.cplx.kernels.simple;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import eth.bram.cplx.GPUContext;
import eth.bram.cplx.GPUDeviceBuffer;
import eth.bram.cplx.GPUKernel;
import eth.bram.cplx.GPUPointer;
import eth.bram.cplx.GPUProgram;
import eth.bram.cplx.cpu.ComplexPolynomial;
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
	 * FFT first step kernel
	 */
	private final GPUKernel fftFirstStepKernel;
	
	/**
	 * FFT kernel
	 */
	private final GPUKernel fftKernel;

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
	private final GPUDeviceBuffer fftIn;
	
	/**
	 * GPU buffer holding in place FFT data
	 */
	private final GPUDeviceBuffer fftOut;
	
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
		
		this.polynomialSize = polynomialSize;
		this.ctx = GPUContext.create(deviceId);
		final GPUProgram program = ctx.createProgram(Arrays.asList(CudaSource.getSource("kernels.cu")));
		final int polynomialItemSize = 2 * Float.BYTES;
		
		// Polynomials are copied in their 2^n form
		this.fftIn = ctx.createDeviceBuffer(polynomialSize * 2 * 2 * polynomialItemSize);
		
		// During preparation for the FFT we will expand the 2^n in 2^(2n) form for multiplication, hence the doubling in size
		this.fftOut = ctx.createDeviceBuffer(polynomialSize * 2 * 2 * polynomialItemSize);
		
		// FFT Kernel and parameters
		this.fftKernel = program.buildKernel("fft");
		fftKernel.setArg(0, fftOut.pointer());
		
		// multiplication kernel
		this.multiplyKernel = program.buildKernel("multiply");
		multiplyKernel.setArg(0, fftOut.pointer());
		multiplyKernel.setArg(1, fftIn.pointer());
		multiplyKernel.setArg(2, polynomialSize * 2);
		
		// FFT Prepare complex kernel and parameters
		this.fftFirstStepKernel = program.buildKernel("fftFirstStep");

		// Finalize multiplication
		this.finalizeMultiplicationKernel = program.buildKernel("finalizeMultiplication");
		finalizeMultiplicationKernel.setArg(0, fftOut.pointer());
		finalizeMultiplicationKernel.setArg(1, fftIn.pointer());
		finalizeMultiplicationKernel.setArg(2, polynomialSize);
	}

	/**
	 * Performs FFT based multiplication of polynomials.
	 * 
	 * @return a A * B.
	 */
	public ComplexPolynomial mul(ComplexPolynomial a, ComplexPolynomial b) {

		final int polynomialCount = 2;
		
		// Check buffer sizes are correct
		checkSize(a.data());
		checkSize(b.data());
		
		// Send polynomials to GPU
		fftIn.write(0, a.data());
		fftIn.write(a.data().capacity() * 2, b.data());
		
		// Prepare data for FFT (2 polynomials of size 2^2n)
		runFFT(fftIn.pointer(), polynomialSize * 2, polynomialCount);
		
		// Perform multiplication on one polynomial size which contains 2 elements (re + im)
		multiplyKernel.run(polynomialSize * 2);
		
		// Run reverse FFT - Only first half (lower powers of n) of it is needed
		runFFT(fftIn.pointer(), polynomialSize * 2, 1);
		
		// Finalize multiplication and compress result. This kernel brings us back from 2n polynomials to n.
		finalizeMultiplicationKernel.run(polynomialSize);
		
		// Read and return result
		return new ComplexPolynomial(fftIn.read(ByteBuffer.allocate(a.data().capacity()).order(ByteOrder.LITTLE_ENDIAN)));
	}

	/**
	 * Runs an FFT on the provided buffer
	 * @param input buffer to run the FFT on
	 * @param itemSize size of each item of the FFT in the provided buffer
	 * @param itemCount number of items to run FFT on in the provided buffer
	 */
	private void runFFT(GPUPointer input, int itemSize, int itemCount) {
		final int rounds = (int)(Math.log(itemSize) / Math.log(2));
		// First step of the FFT
		fftFirstStepKernel.setArg(0, input);
		fftFirstStepKernel.setArg(1, fftOut.pointer());
		fftFirstStepKernel.setArg(2, rounds);
		fftFirstStepKernel.setArg(3, itemSize);
		fftFirstStepKernel.run(itemCount * itemSize / 2, 128);
		// Run FFT kernels for log2 rounds.
		for (int i = 2 ; i <= rounds ; ++i) {
			fftKernel.setArg(1, 1 << i);
			fftKernel.run(itemCount * itemSize / 2, 128);
		}
	}

	/**
	 * Checks the size of the buffer matches the polynomial size provided.
	 */
	private void checkSize(ByteBuffer b) {
		if(b.capacity() != polynomialSize * 2 * Float.BYTES) {
			throw new IllegalArgumentException("Provided buffer does not match the kernel polynomial size");
		}
	}

	/**
	 * Release all created objects in this context : buffers, kernels, streams...
	 */
	public void release() {
		ctx.release();
	}
}