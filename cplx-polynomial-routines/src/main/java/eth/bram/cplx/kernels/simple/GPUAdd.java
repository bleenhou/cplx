package eth.bram.cplx.kernels.simple;

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
 * GPU implementation of the addition of two polynomials
 */
public class GPUAdd {

	/**
	 * GPU context wrapper
	 */
	private final GPUContext ctx;

	/**
	 * Polynomial addition kernel
	 */
	private final GPUKernel add;

	/**
	 * GPU buffer holding A
	 */
	private final GPUBuffer A;

	/**
	 * GPU buffer holding B
	 */
	private final GPUBuffer B;
	
	/**
	 * GPU buffer holding result
	 */
	private final GPUBuffer res;

	/**
	 * polynomial size
	 */
	private final int polynomialSize;
	
	/**
	 * Creates context and buffers on the given device. 
	 * 
	 * @param deviceId index of the cuda device to use.
	 * @param polynomialSize size of the polynomials we work with.
	 */
	public GPUAdd(int deviceId, int polynomialSize) {
		
		// Check polynomial size is a power of two
		if (Integer.bitCount(polynomialSize) != 1) {
			throw new IllegalArgumentException("Polynomial size must be a power of 2");
		}
		
		// Store polynomial size
		this.polynomialSize = polynomialSize;
		
		// Create context and program
		this.ctx = GPUContext.create(deviceId);
		final GPUProgram program = ctx.createProgram(Arrays.asList(CudaSource.getSource("kernels.cu")));
		this.add = program.buildKernel("add");
		
		// Create buffers for A and B. 2 bytes per polynomial element (real and immaginary)
		this.A = 	ctx.createBuffer(ByteBuffer.allocate(polynomialSize * 2 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN));
		this.B = 	ctx.createBuffer(ByteBuffer.allocate(polynomialSize * 2 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN));
		this.res = 	ctx.createBuffer(ByteBuffer.allocate(polynomialSize * 2 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN));
		
		// Set arguments for kernel
		add.setArg(0, A.pointer());
		add.setArg(1, B.pointer());
		add.setArg(2, res.pointer());
	}

	/**
	 * Runs the add kernel, adding two polynomials
	 * 
	 * @return a + b
	 */
	public ComplexPolynomial add(ComplexPolynomial a, ComplexPolynomial p2) {

		// Fill and copy buffers
		A.buffer().put(a.re()).put(a.im()).rewind();
		A.write();
		B.buffer().put(p2.re()).put(p2.im()).rewind();
		B.write();

		// Run kernel. There are (polynomialSize * 2) additions but it's
		// a float2 kernel so number of threads is (polynomialSize)
		add.run(polynomialSize);

		// Read results
		res.read();

		// Create CPU memory for results and return
		return new ComplexPolynomial(res.buffer().slice(0, a.re().capacity()).order(ByteOrder.LITTLE_ENDIAN), 
							 		 res.buffer().slice(a.re().capacity(), a.im().capacity()).order(ByteOrder.LITTLE_ENDIAN));
	}

	/**
	 * Release all created objects in this context : buffers, kernels, streams...
	 */
	public void release() {
		ctx.release();
	}
}
