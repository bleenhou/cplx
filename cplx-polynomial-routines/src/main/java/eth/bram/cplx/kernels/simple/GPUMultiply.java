package eth.bram.cplx.kernels.simple;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import eth.bram.cplx.GPUContext;
import eth.bram.cplx.GPUDeviceBuffer;
import eth.bram.cplx.GPUKernel;
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
	private final GPUKernel fftFirstStepKernelA;
	
	/**
	 * FFT kernel
	 */
	private final GPUKernel fftKernelA;
	
	/**
	 * FFT first step kernel
	 */
	private final GPUKernel fftFirstStepKernelB;
	
	/**
	 * FFT kernel
	 */
	private final GPUKernel fftKernelB;

	/**
	 * Multiply kernel
	 */
	private final GPUKernel multiplyKernel;
	
	/**
	 * Finalization of the multiplication kernel
	 */
	private final GPUKernel finalizeMultiplicationKernel;

	/**
	 * Polynomial A
	 */
	private final GPUDeviceBuffer A;
	
	/**
	 * Polynomial B
	 */
	private final GPUDeviceBuffer B;
	
	/**
	 * Buffer holding in place FFT data
	 */
	private final GPUDeviceBuffer fftA;
	
	/**
	 * Buffer holding in place FFT data
	 */
	private final GPUDeviceBuffer fftB;
	
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
		final GPUProgram program = ctx.createProgram(Arrays.asList(CudaSource.getSource()));
		final int polynomialItemSize = 2 * Float.BYTES;
		
		// Polynomials are copied in their 2^n form
		this.A = ctx.createDeviceBuffer(polynomialSize * 2 * polynomialItemSize);
		this.B = ctx.createDeviceBuffer(polynomialSize * 2 * polynomialItemSize);
		this.fftA = ctx.createDeviceBuffer(polynomialSize * 2 * polynomialItemSize);
		this.fftB = ctx.createDeviceBuffer(polynomialSize * 2 * polynomialItemSize);
		
		// FFT Kernel and parameters
		this.fftFirstStepKernelA = program.buildKernel("fftFirstStep");
		this.fftKernelA = program.buildKernel("fft");
		this.fftFirstStepKernelB = program.buildKernel("fftFirstStep");
		this.fftKernelB = program.buildKernel("fft");
		
		// multiplication kernel
		this.multiplyKernel = program.buildKernel("multiply");
		multiplyKernel.setArg(0, fftA.pointer());
		multiplyKernel.setArg(1, fftB.pointer());
		
		// Finalize multiplication
		this.finalizeMultiplicationKernel = program.buildKernel("finalizeMultiplication");
		finalizeMultiplicationKernel.setArg(0, A.pointer());
		finalizeMultiplicationKernel.setArg(1, fftA.pointer());
		finalizeMultiplicationKernel.setArg(2, polynomialSize);
	}

	/**
	 * Performs FFT based multiplication of polynomials.
	 * 
	 * @return A * B.
	 */
	public ComplexPolynomial mul(ComplexPolynomial a, ComplexPolynomial b) {

		final ExecutorService pool = Executors.newFixedThreadPool(3);
		final Future<?> copyAndFFTA = pool.submit(() -> {
			checkSize(a.data());
			A.write(0, a.data());
			runFFT(A, fftA, fftFirstStepKernelA, fftKernelA, polynomialSize * 2, 1);
		});
		final Future<?> copyAndFFTB = pool.submit(() -> {
			checkSize(b.data());
			B.write(0, b.data());
			runFFT(B, fftB, fftFirstStepKernelB, fftKernelB, polynomialSize * 2, 1);
		});
		final Future<ByteBuffer> allocateResult = pool.submit(() -> {
			return ByteBuffer.allocate(a.data().capacity()).order(ByteOrder.LITTLE_ENDIAN);
		});
		
		ensureFinished(copyAndFFTA);
		ensureFinished(copyAndFFTB);
		
		// Perform multiplication on one extended polynomial
		multiplyKernel.run(polynomialSize * 2);
		
		// Run reverse FFT - Only first half (lower powers of n) of it is needed
		runFFT(fftA, A, fftFirstStepKernelA, fftKernelA, polynomialSize * 2, 1);
		
		// Finalize multiplication and compress result. This kernel brings us back from 2n polynomials to n.
		finalizeMultiplicationKernel.run(polynomialSize);
		
		// Read and return result
		final ByteBuffer result = ensureFinished(allocateResult);
		pool.shutdown();
		return new ComplexPolynomial(fftA.read(result));
	}

	/**
	 * Runs an FFT on the provided buffer
	 * @param input buffer to run the FFT on
	 * @param itemSize size of each item of the FFT in the provided buffer
	 * @param itemCount number of items to run FFT on in the provided buffer
	 */
	private void runFFT(GPUDeviceBuffer input, GPUDeviceBuffer output, GPUKernel firstStep, GPUKernel fft, int itemSize, int itemCount) {
		final int rounds = (int)(Math.log(itemSize) / Math.log(2));
		// First step of the FFT
		firstStep.setArg(0, input.pointer());
		firstStep.setArg(1, output.pointer());
		firstStep.setArg(2, rounds);
		firstStep.setArg(3, itemSize);
		firstStep.run(itemCount * itemSize / 2, 128);
		// Run FFT kernels for log2 rounds.
		for (int i = 2 ; i <= rounds ; ++i) {
			fft.setArg(0, output.pointer());
			fft.setArg(1, 1 << i);
			fft.run(itemCount * itemSize / 2, 128);
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
	
	/**
	 * Waits until a thread is finished.
	 */
	private static <V> V ensureFinished(Future<V> f) {
		try {
			return f.get();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
}