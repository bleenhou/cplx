package eth.bram.cplx.kernels.matrix;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import eth.bram.cplx.ComplexPolynomial;
import eth.bram.cplx.GPUBuffer;
import eth.bram.cplx.GPUContext;
import eth.bram.cplx.GPUKernel;
import eth.bram.cplx.GPUProgram;
import eth.bram.cplx.kernels.CudaSource;

/**
 * GPU matrix multiplication.
 * 
 * This class loops over 4 main steps :
 * 	- Copies a sub portion of matrix B and transforms it's polynomials in FFT representation.
 *  - Copies a sub portion of matrix A and transforms it's polynomials in FFT representation.
 *  - Performs the sub matrix multiplication over the FFT representation.
 *  - Reads results and copy them to CPU.
 *  
 * To accommodate for very large matrices, the size of the sub matrices are configured using maxBufferSizeBytesA and maxBufferSizeBytesB.
 * Typically a bigger maxBufferSizeBytesB will ensure reuse of the B FFT computation results while we iterate on the rows of A.
 * 
 * Copy and executions are performed in parallel. 
 * Each matrix slice is sent to the GPU as soon as previous FFT transformation is finished (and thus the input buffer is available again)
 */
public class GPUMatrixMultiply {

	/**
	 * GPU context wrapper
	 */
	private final GPUContext ctx;

	/**
	 * FFT preparation kernel (Polynomial version, leaves padding)
	 */
	private final GPUKernel fftPreparePolynomialKernelA;
	
	/**
	 * FFT preparation kernel (Polynomial version, leaves padding)
	 */
	private final GPUKernel fftPreparePolynomialKernelB;
	
	/**
	 * FFT kernel for A
	 */
	private final GPUKernel fftKernelA;
	
	/**
	 * FFT kernel for B
	 */
	private final GPUKernel fftKernelB;
	
	/**
	 * FFT kernel for reverse FFT
	 */
	private final GPUKernel fftFistKernelReverse;
	
	/**
	 * FFT kernel for reverse FFT
	 */
	private final GPUKernel fftKernelReverse;

	/**
	 * Multiplication kernel
	 */
	private final GPUKernel multiplyMatrixKernel;
	
	/**
	 * Finalization of the multiplication kernel
	 */
	private final GPUKernel finalizeMultiplicationKernel;
	
	/**
	 * Polynomial size
	 */
	private final int polynomialSize;
	
	/**
	 * Maximum buffer size for matrix A data (for slicing)
	 */
	private final int maxBufferSizeA;
	
	/**
	 * Maximum buffer size for matrix B data (for slicing)
	 */
	private final int maxBufferSizeB;
	
	/**
	 * Creates context and buffers on the given device. 
	 * 
	 * @param deviceId index of the cuda device to use.
	 * @param polynomialSize size of the polynomials we work with.
	 * @param maxBufferSizeA the maximum buffer size to use for matrix A (for slicing).
	 * @param maxBufferSizeB the maximum buffer size to use for matrix B (for slicing).
	 */
	public GPUMatrixMultiply(int deviceId, int polynomialSize, int maxBufferSizeA, int maxBufferSizeB) {
		
		// Check polynomial size is a power of two
		if (Integer.bitCount(polynomialSize) != 1) {
			throw new IllegalArgumentException("Polynomial size must be a power of 2");
		}
		
		// Double polynomial size (from n to 2n to leave room for multiplication)
		this.polynomialSize = polynomialSize * 2;
		this.maxBufferSizeA = maxBufferSizeA;
		this.maxBufferSizeB = maxBufferSizeB;
		
		// Create context and kernels
		// There is one FFT kernel per matrix so that we set arguments once and for all (prevent threading issues setting arguments)
		this.ctx = GPUContext.create(deviceId);
		final GPUProgram program = ctx.createProgram(Arrays.asList(CudaSource.getSource("kernels.cu")));
		this.fftPreparePolynomialKernelA = program.buildKernel("fftPreparePolynomial");
		this.fftPreparePolynomialKernelB = program.buildKernel("fftPreparePolynomial");
		this.fftKernelA = program.buildKernel("fft");
		this.fftKernelB = program.buildKernel("fft");
		this.fftKernelReverse = program.buildKernel("fft");
		this.fftFistKernelReverse = program.buildKernel("fftFirst");
		this.multiplyMatrixKernel = program.buildKernel("multiplyMatrix");
		this.finalizeMultiplicationKernel = program.buildKernel("finalizeMultiplication");

		// Set fixed parameters
		fftFistKernelReverse.setArg(2, rounds());
		fftPreparePolynomialKernelA.setArg(2, rounds());
		fftPreparePolynomialKernelB.setArg(2, rounds());
		finalizeMultiplicationKernel.setArg(2, this.polynomialSize);
		multiplyMatrixKernel.setArg(3, this.polynomialSize);
	}

	/**
	 * Multiplies the matrices A and B
	 */
	public ComplexPolynomial[][] multiply(ComplexPolynomial[][] A, ComplexPolynomial[][] B) {
		
		// Check sizes, matrices must be compatible.
		final int aHeight = A.length;
		final int aWidth = A[0].length;
		final int bHeight = B.length;
		final int bWidth = B[0].length;
		if (aWidth != bHeight || aHeight != bWidth) {
			throw new IllegalArgumentException("Matrices cannot be multiplied");
		}
		
		// Check how many multiplications we can run in parallel
		final int polynomialSizeInBytes = polynomialSize * Float.BYTES;
		final int lineSize = aWidth * polynomialSizeInBytes;
		final int columnSize = lineSize; // alias for readability
		final int rowsPerSlice = Math.min(aHeight, maxBufferSizeA / lineSize);
		final int columnPerSlice = Math.min(bWidth, maxBufferSizeB / columnSize);
		if (rowsPerSlice == 0 || columnPerSlice == 0) {
			throw new IllegalStateException("Window is too small to compute a single result cell");
		}
		
		// Create buffers for polynomials
		final GPUBuffer lines = ctx.createBuffer(ByteBuffer.allocate(lineSize * rowsPerSlice).order(ByteOrder.LITTLE_ENDIAN));
		final GPUBuffer columns = ctx.createBuffer(ByteBuffer.allocate(columnSize * columnPerSlice).order(ByteOrder.LITTLE_ENDIAN));
		fftPreparePolynomialKernelA.setArg(0, lines.pointer());
		fftPreparePolynomialKernelB.setArg(0, columns.pointer());
		
		// Create buffers for FFT Representation of polynomials
		final GPUBuffer AFFT = ctx.createBuffer(ByteBuffer.allocate(lineSize * rowsPerSlice * 2).order(ByteOrder.LITTLE_ENDIAN));
		final GPUBuffer BFFT = ctx.createBuffer(ByteBuffer.allocate(columnSize * columnPerSlice * 2).order(ByteOrder.LITTLE_ENDIAN));
		final GPUBuffer FFTReverse = ctx.createBuffer(ByteBuffer.allocate(rowsPerSlice * columnPerSlice * polynomialSizeInBytes * 2).order(ByteOrder.LITTLE_ENDIAN));
		fftKernelA.setArg(0, AFFT.pointer());
		fftKernelB.setArg(0, BFFT.pointer());
		fftPreparePolynomialKernelA.setArg(1, AFFT.pointer());
		fftPreparePolynomialKernelB.setArg(1, BFFT.pointer());
		fftKernelReverse.setArg(0, FFTReverse.pointer());
		fftFistKernelReverse.setArg(1, FFTReverse.pointer());
		finalizeMultiplicationKernel.setArg(0, FFTReverse.pointer());
		
		// Buffer for multiplication 
		final GPUBuffer ccOut = ctx.createBuffer(ByteBuffer.allocate(rowsPerSlice * columnPerSlice * polynomialSizeInBytes * 2).order(ByteOrder.LITTLE_ENDIAN));
		multiplyMatrixKernel.setArg(0, AFFT.pointer());
		multiplyMatrixKernel.setArg(1, BFFT.pointer());
		multiplyMatrixKernel.setArg(2, ccOut.pointer());
		fftFistKernelReverse.setArg(0, ccOut.pointer());
		
		// Result buffer
		final GPUBuffer result = ctx.createBuffer(ByteBuffer.allocate(rowsPerSlice * columnPerSlice * polynomialSizeInBytes).order(ByteOrder.LITTLE_ENDIAN));
		finalizeMultiplicationKernel.setArg(1, result.pointer());
		
		// A * B Result matrix
		final ComplexPolynomial[][] c = new ComplexPolynomial[aHeight][bWidth];
		
		// Pool to parallelize copy and execution
		final ExecutorService pool = Executors.newFixedThreadPool(4);
		Future<?> resultCopy = CompletableFuture.completedFuture(null);
		
		// Iterate column per column
		Future<?> colCopy = pool.submit(() -> copyColumns(columns, B, 0, Math.min(columnPerSlice, bWidth)));
		for (int i = 0 ; i < bWidth ; i += Math.min(bWidth - i, columnPerSlice)) {
			
			// Number of columns this round
			final int columnCount = Math.min(i + columnPerSlice, bWidth) - i;
			
			// Start column FFT as soon as column copy is finished
			ensureFinished(colCopy);
			final Future<?> fftColumns = pool.submit(() -> fftColumns(columns, B));
			
			// Start copying data for the next column iteration
			final int startColumn = i;
			colCopy = pool.submit(() -> {
				ensureFinished(fftColumns); // Copy only when FFT is finished
				final int from = startColumn + columnPerSlice;
				final int to = from + columnPerSlice;
				copyColumns(columns, B, Math.min(from, bWidth), Math.min(to, bWidth));
			});
			
			// Copy the first row asynchronously
			Future<?> rowCopy = pool.submit(() -> copyRows(lines, A, 0, Math.min(rowsPerSlice, aHeight)));
			for (int j = 0 ; j < aHeight ; j += Math.min(aHeight - j, rowsPerSlice)) {
				
				// Number of rows this round
				final int rowCount = Math.min(j + rowsPerSlice, aHeight) - j;
				
				// Start FFT on the row as soon as copy is finished
				ensureFinished(rowCopy); 
				fftRows(lines, A);
				
				// Start copying data for the next row iteration
				final int startRow = j;
				rowCopy = pool.submit(() -> {
					final int from = startRow + rowsPerSlice;
					final int to = from + rowsPerSlice;
					copyRows(lines, A, Math.min(from, aHeight), Math.min(to, aHeight));
				});
				
				// Make sure columns have been FFT'ed before starting the multiplication
				ensureFinished(fftColumns);
				
				// Perform multiplication on this section (columnCount * rowCount polynomial multiplications)
				multiplyMatrixKernel.setArg(4, columnCount);
				multiplyMatrixKernel.setArg(5, rowCount);
				multiplyMatrixKernel.setArg(6, aWidth);
				multiplyMatrixKernel.run(columnCount * rowCount * polynomialSize);
				
				// Reverse FFT on the result of the multiplication
				final int rounds = rounds();
				fftFistKernelReverse.run(columnCount * rowCount * polynomialSize / 2);
				for (int k = 2 ; k <= rounds ; ++k) {
					fftKernelReverse.setArg(1, 1 << k);
					fftKernelReverse.run(columnCount * rowCount * polynomialSize / 2);
				}
				
				// Finalize multiplication and compress result
				ensureFinished(resultCopy); // Make sure results of previous slice have been retrieved before overriding buffer
				finalizeMultiplicationKernel.run(columnCount * rowCount * polynomialSize / 2);
				
				// Read results and copy back asynchronously
				resultCopy = pool.submit(() -> readResults(polynomialSizeInBytes, c, startColumn, columnCount, startRow, rowCount, result));
			}
		}
	
		ensureFinished(resultCopy);
		pool.shutdown();
		return c;
	}

	/**
	 * Reads results and copy objects to CPU.
	 */
	private void readResults(int polynomialSizeInBytes, ComplexPolynomial[][] cpuResult, int startColumn, int columnCount, int startRow, int rowCount, GPUBuffer result) {
		result.read();
		int pos = 0;
		for (int c = startRow ; c < startRow + rowCount ; ++c) {
			for (int r = startColumn ; r < startColumn + columnCount ; ++r) {
				final ByteBuffer re = ByteBuffer.allocate(polynomialSizeInBytes / 2).order(ByteOrder.LITTLE_ENDIAN);
				final ByteBuffer im = ByteBuffer.allocate(polynomialSizeInBytes / 2).order(ByteOrder.LITTLE_ENDIAN);
				re.put(result.buffer().slice(pos, polynomialSizeInBytes / 2));
				im.put(result.buffer().slice(pos + polynomialSizeInBytes / 2, polynomialSizeInBytes / 2));
				cpuResult[c][r] = new ComplexPolynomial(re, im);
				pos += polynomialSizeInBytes;
			}
		}
	}
	
	/**
	 * Runs FFT on the data in the columns buffer
	 */
	private void fftColumns(GPUBuffer polynomials, ComplexPolynomial[][] p) {
		// Prepare data for FFT for the polynomials
		final int itemCount = polynomials.buffer().position() / Float.BYTES;
		fftPreparePolynomialKernelB.run(itemCount);
		
		// Run FFT kernels on all polynomials, log2(polynomialSize) rounds.
		final int rounds = rounds();
		for (int i = 1 ; i <= rounds ; ++i) {
			fftKernelB.setArg(1, 1 << i);
			fftKernelB.run(itemCount / 2);
		}
	}
	
	/**
	 * Runs FFT on the rows buffer
	 */
	private void fftRows(GPUBuffer polynomials, ComplexPolynomial[][] p) {
		// Prepare data for FFT for the polynomials
		final int itemCount = polynomials.buffer().position() / Float.BYTES;
		fftPreparePolynomialKernelA.run(itemCount);
		
		// Run FFT kernels on all polynomials, log2(polynomialSize) rounds.
		final int rounds = rounds();
		for (int i = 1 ; i <= rounds ; ++i) {
			fftKernelA.setArg(1, 1 << i);
			fftKernelA.run(itemCount / 2);
		}
	}
	
	/**
	 * Columnar copy of the B matrix at the specified indexes
	 */
	private void copyColumns(GPUBuffer polynomials, ComplexPolynomial[][] p, int from, int to) {
		if (from == to) {
			return;
		}
		polynomials.buffer().rewind();
		for (int column = from; column < to ; ++column) {
			for (int i = 0 ; i < p.length ; ++i) {
				polynomials.buffer().put(p[i][column].re().rewind()).put(p[i][column].im().rewind());
			}
		}
		polynomials.write();
	}

	/**
	 * Linear copy of the A matrix at the specified indexes
	 */
	private void copyRows(GPUBuffer polynomials, ComplexPolynomial[][] p, int from, int to) {
		if (from == to) {
			return;
		}
		polynomials.buffer().rewind();
		for (int row = from; row < to ; ++row) {
			for (int i = 0 ; i < p[row].length ; ++i) {
				polynomials.buffer().put(p[row][i].re().rewind()).put(p[row][i].im().rewind());
			}
		}
		polynomials.write();
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
