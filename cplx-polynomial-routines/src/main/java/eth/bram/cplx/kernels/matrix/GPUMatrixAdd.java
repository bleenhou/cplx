package eth.bram.cplx.kernels.matrix;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import eth.bram.cplx.GPUBuffer;
import eth.bram.cplx.GPUContext;
import eth.bram.cplx.GPUKernel;
import eth.bram.cplx.GPUProgram;
import eth.bram.cplx.cpu.ComplexPolynomial;
import eth.bram.cplx.kernels.CudaSource;

/**
 * GPU implementation of the addition of two matrixes of complex polynomials.
 * 
 * To accommodate for very large matrices, the size of the sub computation is configured using maxBufferSize.
 */
public class GPUMatrixAdd {

	/**
	 * GPU context wrapper
	 */
	private final GPUContext ctx;

	/**
	 * Polynomial addition kernel
	 */
	private final GPUKernel addKernel;

	/**
	 * Polynomial size
	 */
	private final int polynomialSize;
	
	/**
	 * Maximum buffer size for matrix A data (for slicing)
	 */
	private final int maxBufferSize;
	
	/**
	 * Creates context and buffers on the given device. 
	 * 
	 * @param deviceId index of the cuda device to use.
	 * @param polynomialSize size of the polynomials we work with.
	 * @param maxBufferSize the maximum buffer size to use for slicing.
	 */
	public GPUMatrixAdd(int deviceId, int polynomialSize, int maxBufferSize) {
		
		// Check polynomial size is a power of two
		if (Integer.bitCount(polynomialSize) != 1) {
			throw new IllegalArgumentException("Polynomial size must be a power of 2");
		}
		
		this.polynomialSize = polynomialSize;
		this.maxBufferSize = maxBufferSize;
		
		// Create GPU context and kernels
		this.ctx = GPUContext.create(deviceId);
		final GPUProgram program = ctx.createProgram(Arrays.asList(CudaSource.getSource()));
		this.addKernel = program.buildKernel("add");
	}

	/**
	 * Adds the matrices A and B
	 */
	public ComplexPolynomial[][] add(ComplexPolynomial[][] A, ComplexPolynomial[][] B) {
		
		// Check sizes, matrices must be compatible.
		final int aHeight = A.length;
		final int aWidth = A[0].length;
		final int bHeight = B.length;
		final int bWidth = B[0].length;
		if (aWidth != bWidth || aHeight != bHeight) {
			throw new IllegalArgumentException("Matrices cannot be added");
		}
		
		// Check how many additions we can run in parallel (line per line)
		final int polynomialSizeInBytes = polynomialSize * 2 * Float.BYTES;
		final int lineSize = aWidth * polynomialSizeInBytes;
		final int rows = Math.min(aHeight, maxBufferSize / lineSize);
		if (rows == 0) {
			throw new IllegalStateException("Window is too small to compute a single result row");
		}
		
		// Create buffers for polynomials
		final GPUBuffer linesA = ctx.createBuffer(ByteBuffer.allocate(lineSize * rows).order(ByteOrder.LITTLE_ENDIAN));
		final GPUBuffer linesB = ctx.createBuffer(ByteBuffer.allocate(lineSize * rows).order(ByteOrder.LITTLE_ENDIAN));
		final GPUBuffer res = ctx.createBuffer(ByteBuffer.allocate(lineSize * rows).order(ByteOrder.LITTLE_ENDIAN));
		addKernel.setArg(0, linesA.pointer());
		addKernel.setArg(1, linesB.pointer());
		addKernel.setArg(2, res.pointer());
		
		// A + B Result matrix
		final ComplexPolynomial[][] c = new ComplexPolynomial[aHeight][bWidth];
		
		// Pool to parallelize buffer creation and copies
		final ExecutorService pool = Executors.newFixedThreadPool(2);
		
		// Iterate block by block
		for (int i = 0 ; i < aHeight ; i += Math.min(aHeight - i, rows)) {
			
			final int from = i;
			final int to = i + Math.min(aHeight - i, rows);
			final int rowCount = to - from;
			final Future<?> copyA = pool.submit(() -> copyRows(linesA, A, from, to));
			final Future<?> copyB = pool.submit(() -> copyRows(linesB, B, from, to));
			
			// Perform addition once copies are finished.
			ensureFinished(copyA);
			ensureFinished(copyB);
			addKernel.run(rowCount * aWidth * polynomialSize);
			
			// Read and copy results
			res.read();
			int pos = 0;
			for (int j = from ; j < to ; ++j) {
				for (int k = 0 ; k < A[j].length ; ++k) {
					final ByteBuffer data = ByteBuffer.allocate(polynomialSizeInBytes).order(ByteOrder.LITTLE_ENDIAN);
					data.put(res.buffer().slice(pos, polynomialSizeInBytes));
					c[j][k] = new ComplexPolynomial(data);
					pos += polynomialSizeInBytes;
				}
			}
		}
	
		pool.shutdown();
		return c;
	}

	/**
	 * Copy the selected rows in the provided buffer
	 */
	private void copyRows(GPUBuffer b, ComplexPolynomial[][] A, int from, int to) {
		b.buffer().rewind();
		for (int i = from ; i < to ; ++i) {
			for (int j = 0 ; j < A[i].length ; ++j) {
				b.buffer().put(A[i][j].data());
			}
		}
		b.write();
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
	
	/**
	 * Release all created objects in this context : buffers, kernels, streams...
	 */
	public void release() {
		ctx.release();
	}
}
