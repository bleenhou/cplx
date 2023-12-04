package eth.bram.cplx.kernels.matrix;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;

import eth.bram.cplx.GPUBuffer;
import eth.bram.cplx.GPUContext;
import eth.bram.cplx.GPUKernel;
import eth.bram.cplx.GPUProgram;
import eth.bram.cplx.cpu.ComplexPolynomial;
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
	 * MAX_ROWS to process in parallel.
	 */
	private static final int MAX_ROWS = 8;
	
	/**
	 * GPU context wrapper
	 */
	private final GPUContext ctx;

	/**
	 * FFT preparation kernel
	 */
	private final GPUKernel fftFirstStepKernelA;
	
	/**
	 * FFT preparation kernel
	 */
	private final GPUKernel fftFirstStepKernelB;
	
	/**
	 * FFT preparation kernel
	 */
	private final GPUKernel fftFirstStepKernelReverse;
	
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
		
		this.polynomialSize = polynomialSize;
		this.maxBufferSizeA = maxBufferSizeA;
		this.maxBufferSizeB = maxBufferSizeB;
		
		// Create context and kernels
		// There is one FFT kernel per matrix so that we set arguments once and for all (prevent threading issues setting arguments)
		this.ctx = GPUContext.create(deviceId);
		final GPUProgram program = ctx.createProgram(Arrays.asList(CudaSource.getSource("kernels.cu")));
		this.fftFirstStepKernelA = program.buildKernel("fftFirstStep");
		this.fftFirstStepKernelB = program.buildKernel("fftFirstStep");
		this.fftFirstStepKernelReverse = program.buildKernel("fftFirstStep");
		this.fftKernelA = program.buildKernel("fft");
		this.fftKernelB = program.buildKernel("fft");
		this.fftKernelReverse = program.buildKernel("fft");
		this.multiplyMatrixKernel = program.buildKernel("multiplyMatrix");
		this.finalizeMultiplicationKernel = program.buildKernel("finalizeMultiplication");

		// Set fixed parameters
		multiplyMatrixKernel.setArg(3, this.polynomialSize * 2);
		finalizeMultiplicationKernel.setArg(2, this.polynomialSize);
	}

	private static final AtomicLong FFTA_TIMER = new AtomicLong(0L);
	private static final AtomicLong FFTB_TIMER = new AtomicLong(0L);
	private static final AtomicLong FFTREV_TIMER = new AtomicLong(0L);
	private static final AtomicLong MUL_TIMER = new AtomicLong(0L);
	
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
		final int polynomialSizeInBytes = polynomialSize * 2 * Float.BYTES;
		final int lineSize = aWidth * polynomialSizeInBytes * 2;
		final int columnSize = lineSize; // alias for readability
		final int rowsPerSlice = Math.min(Math.min(aHeight, maxBufferSizeA / lineSize), MAX_ROWS);
		final int columnPerSlice = Math.min(bWidth, maxBufferSizeB / columnSize);
		if (rowsPerSlice == 0 || columnPerSlice == 0) {
			throw new IllegalStateException("Window is too small to compute a single result cell");
		}
		
		// Create buffers for polynomials - we use twice the space to leave room for polynomials of size 2n
		final GPUBuffer lines = ctx.createBuffer(ByteBuffer.allocate(lineSize * rowsPerSlice).order(ByteOrder.LITTLE_ENDIAN));
		final GPUBuffer columns = ctx.createBuffer(ByteBuffer.allocate(columnSize * columnPerSlice).order(ByteOrder.LITTLE_ENDIAN));
		
		// Create buffers for FFT Representation of polynomials
		final GPUBuffer AFFT = ctx.createBuffer(ByteBuffer.allocate(lineSize * rowsPerSlice).order(ByteOrder.LITTLE_ENDIAN));
		final GPUBuffer BFFT = ctx.createBuffer(ByteBuffer.allocate(columnSize * columnPerSlice).order(ByteOrder.LITTLE_ENDIAN));
		final GPUBuffer FFTReverse = ctx.createBuffer(ByteBuffer.allocate(rowsPerSlice * columnPerSlice * polynomialSizeInBytes * 2).order(ByteOrder.LITTLE_ENDIAN));
		fftKernelA.setArg(0, AFFT.pointer());
		fftKernelB.setArg(0, BFFT.pointer());
		finalizeMultiplicationKernel.setArg(0, FFTReverse.pointer());
		
		// Buffer for multiplication 
		final GPUBuffer ccOut = ctx.createBuffer(ByteBuffer.allocate(rowsPerSlice * columnPerSlice * polynomialSizeInBytes * 2).order(ByteOrder.LITTLE_ENDIAN));
		multiplyMatrixKernel.setArg(0, AFFT.pointer());
		multiplyMatrixKernel.setArg(1, BFFT.pointer());
		multiplyMatrixKernel.setArg(2, ccOut.pointer());
		fftKernelReverse.setArg(0, FFTReverse.pointer());
		
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
			final Future<?> fftColumns = pool.submit(() -> fftColumns(columns, BFFT));
			
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
				fftRows(lines, AFFT);
				
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
				final int polynomialCount = columnCount * rowCount;
				multiplyMatrixKernel.setArg(4, columnCount);
				multiplyMatrixKernel.setArg(5, rowCount);
				multiplyMatrixKernel.setArg(6, aWidth);
				long then = System.nanoTime();
				multiplyMatrixKernel.run(columnCount * polynomialSize * 2, 128);
				MUL_TIMER.addAndGet(System.nanoTime() - then);
				
				// Reverse FFT
				then = System.nanoTime();
				runFFT(fftFirstStepKernelReverse, fftKernelReverse, ccOut, FFTReverse, polynomialSize * 2, polynomialCount);
				FFTREV_TIMER.addAndGet(System.nanoTime() - then);
				
				// Finalize multiplication and compress result
				ensureFinished(resultCopy); // Make sure results of previous slice have been retrieved before overriding buffer
				finalizeMultiplicationKernel.run(polynomialCount * polynomialSize);
				
				// Read results and copy back asynchronously
				resultCopy = pool.submit(() -> readResults(polynomialSizeInBytes, c, startColumn, columnCount, startRow, rowCount, result));
			}
		}
	
		ensureFinished(resultCopy);
		
		System.out.println();
		System.out.println("FFT(A) : " + FFTA_TIMER.get() / 1000000 + " ms");
		System.out.println("FFT(B) : " + FFTB_TIMER.get() / 1000000 + " ms");
		System.out.println("FFT-1(A*B) : " + FFTREV_TIMER.get() / 1000000 + " ms");
		System.out.println("MUL : " + MUL_TIMER.get() / 1000000 + " ms");
		System.out.println();
		
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
				final ByteBuffer data = ByteBuffer.allocate(polynomialSizeInBytes).order(ByteOrder.LITTLE_ENDIAN);
				data.put(result.buffer().slice(pos, polynomialSizeInBytes));
				cpuResult[c][r] = new ComplexPolynomial(data);
				pos += polynomialSizeInBytes;
			}
		}
	}
	
	private void runFFT(GPUKernel kPrepare, GPUKernel kRun, GPUBuffer input, GPUBuffer output, int itemSize, int itemCount) {
		final int rounds = (int)(Math.log(itemSize) / Math.log(2));
		// First step of the FFT
		kPrepare.setArg(0, input.pointer());
		kPrepare.setArg(1, output.pointer());
		kPrepare.setArg(2, rounds);
		kPrepare.setArg(3, itemSize);
		kPrepare.run(itemCount * itemSize / 2, 128);
		// Run FFT kernels for log2 rounds.
		for (int i = 2 ; i <= rounds ; ++i) {
			kRun.setArg(1, 1 << i);
			kRun.run(itemCount * itemSize / 2, 128);
		}
	}
	
	/**
	 * Runs FFT on the data in the columns buffer
	 */
	private void fftColumns(GPUBuffer polynomials, GPUBuffer BFFT) {
		final int polynomialCount = polynomials.buffer().position() / (polynomialSize * Float.BYTES * 4);
		long then = System.nanoTime();
		runFFT(fftFirstStepKernelB, fftKernelB, polynomials, BFFT, polynomialSize * 2, polynomialCount);
		FFTB_TIMER.addAndGet(System.nanoTime() - then);
	}
	
	/**
	 * Runs FFT on the rows buffer
	 */
	private void fftRows(GPUBuffer polynomials, GPUBuffer AFFT) {
		final int polynomialCount = polynomials.buffer().position() / (polynomialSize * Float.BYTES * 4);
		long then = System.nanoTime();
		runFFT(fftFirstStepKernelA, fftKernelA, polynomials, AFFT, polynomialSize * 2, polynomialCount); 
		FFTA_TIMER.addAndGet(System.nanoTime() - then);
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
				// Position buffer to leave gaps
				final ByteBuffer data = p[i][column].data();
				polynomials.buffer().put(data.rewind());
				polynomials.buffer().position(polynomials.buffer().position() + data.capacity());
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
				final ByteBuffer data = p[row][i].data().rewind();
				polynomials.buffer().put(data.rewind());
				polynomials.buffer().position(polynomials.buffer().position() + data.capacity());
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
	 * Release all created objects in this context : buffers, kernels, streams...
	 */
	public void release() {
		ctx.release();
	}
}