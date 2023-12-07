package eth.bram.cplx.kernels.matrix;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;

import eth.bram.cplx.GPUBuffer;
import eth.bram.cplx.GPUContext;
import eth.bram.cplx.GPUDeviceBuffer;
import eth.bram.cplx.GPUKernel;
import eth.bram.cplx.GPUPointer;
import eth.bram.cplx.GPUProgram;
import eth.bram.cplx.Utils;
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
	 * MAX_ROWS to process in parallel during matrix multiply
	 */
	private static final int MAX_ROWS = 16;
	
	/**
	 * Local size to use for matrix multiply. Has an impact on the shared memory used.
	 */
	private static final int LOCAL_SIZE_MATRIX_MULTIPLY = 128;
	
	/**
	 * Number of threads to use to copy polynomials to GPU
	 */
	private static final int COPY_THREADS = Runtime.getRuntime().availableProcessors();
	
	/**
	 * GPU context wrapper
	 */
	private final GPUContext ctx;

	/**
	 * FFT preparation kernel
	 */
	private final GPUKernel fftFirstStepKernel;
	
	/**
	 * FFT kernel for A
	 */
	private final GPUKernel fftKernel;
	
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
	private final long maxBufferSizeA;
	
	/**
	 * Maximum buffer size for matrix B data (for slicing)
	 */
	private final long maxBufferSizeB;
	
	/**
	 * Creates context and buffers on the given device. 
	 * 
	 * @param deviceId index of the cuda device to use.
	 * @param polynomialSize size of the polynomials we work with.
	 * @param maxBufferSizeA the maximum buffer size to use for matrix A (for slicing).
	 * @param maxBufferSizeB the maximum buffer size to use for matrix B (for slicing).
	 */
	public GPUMatrixMultiply(int deviceId, int polynomialSize, long maxBufferSizeA, long maxBufferSizeB) {
		
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
		final GPUProgram program = ctx.createProgram(Arrays.asList(CudaSource.getSource(LOCAL_SIZE_MATRIX_MULTIPLY)));
		this.fftFirstStepKernel = program.buildKernel("fftFirstStep");
		this.fftKernel = program.buildKernel("fft");
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
	
	private static final AtomicLong COPY_A_TIMER = new AtomicLong(0L);
	private static final AtomicLong COPY_B_TIMER = new AtomicLong(0L);
	private static final AtomicLong COPY_RES_TIMER = new AtomicLong(0L);
	
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
		final int widePolynomialSizeInBytes = polynomialSizeInBytes * 2;
		final long lineSize = aWidth * widePolynomialSizeInBytes;
		final long columnSize = lineSize; // alias for readability
		final int rowsPerSlice = (int) Math.min(Math.min(aHeight, maxBufferSizeA / lineSize), MAX_ROWS);
		final int columnPerSlice = (int) Math.min(bWidth, maxBufferSizeB / columnSize);
		if (rowsPerSlice == 0 || columnPerSlice == 0) {
			throw new IllegalStateException("Window is too small to compute a single result cell");
		}
		
		long then = System.nanoTime();
		
		// Create buffers for polynomials - we use twice the space to leave room for polynomials of size 2n
		final GPUDeviceBuffer lines = ctx.createDeviceBuffer(lineSize * rowsPerSlice);
		final GPUDeviceBuffer columns = ctx.createDeviceBuffer(columnSize * columnPerSlice);
		
		// Create buffers for FFT Representation of polynomials
		final GPUDeviceBuffer AFFT = ctx.createDeviceBuffer(lineSize * rowsPerSlice);
		final GPUDeviceBuffer BFFT = ctx.createDeviceBuffer(columnSize * columnPerSlice);
		final GPUDeviceBuffer FFTReverse = ctx.createDeviceBuffer(rowsPerSlice * columnPerSlice * widePolynomialSizeInBytes);
		finalizeMultiplicationKernel.setArg(0, FFTReverse.pointer());
		
		// Buffer for multiplication 
		final GPUDeviceBuffer ccOut = ctx.createDeviceBuffer(rowsPerSlice * columnPerSlice * widePolynomialSizeInBytes);
		multiplyMatrixKernel.setArg(0, AFFT.pointer());
		multiplyMatrixKernel.setArg(1, BFFT.pointer());
		multiplyMatrixKernel.setArg(2, ccOut.pointer());
		
		// Result buffer
		final GPUBuffer result = ctx.createBuffer(ByteBuffer.allocate(rowsPerSlice * columnPerSlice * polynomialSizeInBytes).order(ByteOrder.LITTLE_ENDIAN));
		finalizeMultiplicationKernel.setArg(1, result.pointer());
		
		System.out.println("Allocate buffers : " + (System.nanoTime() - then) / 1000000 + " ms");
		
		// A * B Result matrix
		final ComplexPolynomial[][] c = new ComplexPolynomial[aHeight][bWidth];
		
		// Pool to parallelize copy and execution
		final ExecutorService bufferCreationPool = Executors.newFixedThreadPool(COPY_THREADS);
		final ArrayBlockingQueue<ByteBuffer> bufferPool = new ArrayBlockingQueue<>(COPY_THREADS);
		for (int i = 0 ; i < COPY_THREADS ; ++i) {
			bufferPool.add(ByteBuffer.allocate((int) lineSize));
		}
		
		// Copy first column async
		final ExecutorService pool = Executors.newFixedThreadPool(3);
		Future<?> colCopy = pool.submit(() -> copyColumns(bufferCreationPool, bufferPool, columns, B, 0, Math.min(columnPerSlice, bWidth)));
		Future<?> resultCopy = CompletableFuture.completedFuture(null);
		
		// Iterate columns per columns
		for (int i = 0 ; i < bWidth ; i += Math.min(bWidth - i, columnPerSlice)) {
			
			// Copy the row async
			Future<?> rowCopy = pool.submit(() -> copyRows(bufferCreationPool, bufferPool,lines, A, 0, Math.min(rowsPerSlice, aHeight)));
			
			// Number of columns this round
			final int columnCount = Math.min(i + columnPerSlice, bWidth) - i;
			
			// Start column FFT as soon as column copy is finished
			then = System.nanoTime();
			Utils.ensureFinished(colCopy);
			COPY_B_TIMER.addAndGet(System.nanoTime() - then);
			
			then = System.nanoTime();
			runFFT(fftFirstStepKernel, fftKernel, columns.pointer(), BFFT.pointer(), polynomialSize * 2, columnCount * bHeight);
			FFTB_TIMER.addAndGet(System.nanoTime() - then);
			
			final int startColumn = i;
			colCopy = pool.submit(() -> copyColumns(bufferCreationPool, bufferPool, columns, B, Math.min(startColumn + columnPerSlice, bWidth), Math.min(startColumn + columnPerSlice * 2, bWidth)));
			
			// Iterate rows per rows
			for (int j = 0 ; j < aHeight ; j += Math.min(aHeight - j, rowsPerSlice)) {
				
				// Number of rows this round
				final int rowCount = Math.min(j + rowsPerSlice, aHeight) - j;
				
				// Start FFT on the row as soon as copy is finished
				then = System.nanoTime();
				Utils.ensureFinished(rowCopy); 
				COPY_A_TIMER.addAndGet(System.nanoTime() - then);
				
				then = System.nanoTime();
				runFFT(fftFirstStepKernel, fftKernel, lines.pointer(), AFFT.pointer(), polynomialSize * 2, rowCount * aWidth); 
				FFTA_TIMER.addAndGet(System.nanoTime() - then);
				
				// Start copying data for the next row iteration
				final int startRow = j;
				rowCopy = pool.submit(() -> copyRows(bufferCreationPool, bufferPool, lines, A, Math.min(startRow + rowsPerSlice, aHeight), Math.min(startRow + rowsPerSlice * 2, aHeight)));
				
				// Perform multiplication on this section (columnCount * rowCount polynomial multiplications)
				final int polynomialCount = columnCount * rowCount;
				multiplyMatrixKernel.setArg(4, columnCount);
				multiplyMatrixKernel.setArg(5, rowCount);
				multiplyMatrixKernel.setArg(6, aWidth);
				then = System.nanoTime();
				multiplyMatrixKernel.run(columnCount * polynomialSize * 2, LOCAL_SIZE_MATRIX_MULTIPLY);
				MUL_TIMER.addAndGet(System.nanoTime() - then);
				
				// Reverse FFT
				then = System.nanoTime();
				runFFT(fftFirstStepKernel, fftKernel, ccOut.pointer(), FFTReverse.pointer(), polynomialSize * 2, polynomialCount);
				FFTREV_TIMER.addAndGet(System.nanoTime() - then);
				
				// Finalize multiplication and compress result
				Utils.ensureFinished(resultCopy); // Make sure results of previous slice have been retrieved before overriding buffer
				finalizeMultiplicationKernel.run(polynomialCount * polynomialSize, 128);
				
				// Read results and copy back asynchronously
				resultCopy = pool.submit(() -> readResults(polynomialSizeInBytes, c, startColumn, columnCount, startRow, rowCount, result));
			}
		}
	
		Utils.ensureFinished(resultCopy);
		
		System.out.println();
		System.out.println("FFT(A) : " + FFTA_TIMER.get() / 1000000 + " ms");
		System.out.println("FFT(B) : " + FFTB_TIMER.get() / 1000000 + " ms");
		System.out.println("FFT-1(A*B) : " + FFTREV_TIMER.get() / 1000000 + " ms");
		System.out.println("MUL : " + MUL_TIMER.get() / 1000000 + " ms");
		System.out.println();
		System.out.println("WAIT(COPY(A)) : " + COPY_A_TIMER.get() / 1000000 + " ms");
		System.out.println("WAIT(COPY(B)) : " + COPY_B_TIMER.get() / 1000000 + " ms");
		System.out.println("WAIT(READ(RES)) : " + COPY_RES_TIMER.get() / 1000000 + " ms");

		pool.shutdown();
		bufferCreationPool.shutdown();
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
	
	private void runFFT(GPUKernel kPrepare, GPUKernel kRun, GPUPointer input, GPUPointer output, int itemSize, int itemCount) {
		final int rounds = (int)(Math.log(itemSize) / Math.log(2));
		// First steps of the FFT - up to 6 rounds
		kPrepare.setArg(0, input);
		kPrepare.setArg(1, output);
		kPrepare.setArg(2, rounds);
		kPrepare.setArg(3, itemSize);
		kPrepare.run(itemCount * itemSize / 2, 128);
		// Run remaining FFT kernels for up to log2(size) - 6.
		kRun.setArg(0, output);
		for (int i = 7 ; i <= rounds ; ++i) {
			kRun.setArg(1, 1 << i);
			kRun.run(itemCount * itemSize / 2, 128);
		}
	}
	
	/**
	 * Columnar copy of the B matrix at the specified indexes
	 */
	private void copyColumns(ExecutorService exc, ArrayBlockingQueue<ByteBuffer> bufferPool, GPUDeviceBuffer polynomials, ComplexPolynomial[][] p, int from, int to) {
		if (from == to) {
			return;
		}
		final List<Future<?>> futures = new ArrayList<>((to - from));
		for (int c = from; c < to ; ++c) {
			final int col = c;
			final long size = p.length * p[0][col].data().capacity() * 2;
			final long offset = (c - from) * size;
			futures.add(exc.submit(() -> {
				final ByteBuffer set = Utils.safeTake(bufferPool);
				for (int i = 0 ; i < p.length ; ++i) {
					final ByteBuffer data = p[i][col].data().rewind();
					set.put(data).position(set.position() + data.capacity());
				}
				polynomials.write(offset, set.rewind());
				Utils.safePut(bufferPool, set);
			}));
		}
		futures.forEach(Utils::ensureFinished);
	}

	/**
	 * Linear copy of the A matrix at the specified indexes
	 */
	private void copyRows(ExecutorService exc, ArrayBlockingQueue<ByteBuffer> bufferPool, GPUDeviceBuffer polynomials, ComplexPolynomial[][] p, int from, int to) {
		if (from == to) {
			return;
		}
		final List<Future<?>> futures = new ArrayList<>((to - from));
		for (int r = from; r < to ; ++r) {
			final int row = r;
			final long size = p[r].length * p[row][0].data().capacity() * 2;
			final long offset = (r - from) * size;
			futures.add(exc.submit(() -> {
				final ByteBuffer set = Utils.safeTake(bufferPool);
				for (int i = 0 ; i < p[row].length ; ++i) {
					final ByteBuffer data = p[row][i].data().rewind();
					set.put(data).position(set.position() + data.capacity());
				}
				polynomials.write(offset, set.rewind());
				Utils.safePut(bufferPool, set);
			}));
		}
		futures.forEach(Utils::ensureFinished);
	}

	
	/**
	 * Release all created objects in this context : buffers, kernels, streams...
	 */
	public void release() {
		ctx.release();
	}
}