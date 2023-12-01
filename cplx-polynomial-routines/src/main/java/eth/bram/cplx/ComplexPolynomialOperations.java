package eth.bram.cplx;

import eth.bram.cplx.cpu.ComplexPolynomial;
import eth.bram.cplx.kernels.matrix.GPUMatrixAdd;
import eth.bram.cplx.kernels.matrix.GPUMatrixMultiply;
import eth.bram.cplx.kernels.simple.GPUAdd;
import eth.bram.cplx.kernels.simple.GPUMultiply;

/**
 * Entry point for GPU implementation of the operations.
 * This class handles kernel and all GPU resources lifecycle.
 */
public class ComplexPolynomialOperations {

	/**
	 * Kernel for GPU addition.
	 */
	private final GPUAdd add;
	
	/**
	 * Kernel for GPU multiplication.
	 */
	private final GPUMultiply multiply;
	
	/**
	 * Kernel for GPU matrix addition.
	 */
	private final GPUMatrixAdd matrixAdd;
	
	/**
	 * Kernel for GPU matrix multiplication.
	 */
	private final GPUMatrixMultiply matrixMultiply;
	
	/**
	 * Creates all kernels for the associated polynomial size.
	 * @param polynomialSize size of the polynomials this instance will handle.
	 * @param sliceSizeA if matrix computations have to be sliced, max buffer size (in bytes) for matrix A.
	 * @param sliceSizeB if matrix computations have to be sliced, max buffer size (in bytes) for matrix B.
	 */
	public ComplexPolynomialOperations(int polynomialSize, int sliceSizeA, int sliceSizeB) {
		this.add = new GPUAdd(0, polynomialSize);
		this.multiply = new GPUMultiply(0, polynomialSize);
		this.matrixAdd = new GPUMatrixAdd(0, polynomialSize, sliceSizeB);
		this.matrixMultiply = new GPUMatrixMultiply(0, polynomialSize, sliceSizeA, sliceSizeB); 
	}
	
	/**
	 * Creates all kernels for the associated polynomial size. Default values for slicing are used.
	 * @param polynomialSize size of the polynomials this instance will handle.
	 */
	public ComplexPolynomialOperations(int polynomialSize) {
		// Default window size is 128Mb / 960Mb
		// 960Mb is not random. It ensures multiply kernel can run on a 96 local size when windowing, ensuring the fastest memory throughput
		this(polynomialSize, 128 * 1024 * 1024, 960 * 1024 * 1024); 
	}
	
	/**
	 * Add two {@link ComplexPolynomial}.
	 */
	public ComplexPolynomial add(ComplexPolynomial p1, ComplexPolynomial p2) {
		return add.add(p1, p2);
	}
	
	/**
	 * Multiply two {@link ComplexPolynomial}.
	 */
	public ComplexPolynomial multiply(ComplexPolynomial p1, ComplexPolynomial p2) {
		return  multiply.mul(p1, p2);
	}
	
	/**
	 * Add two {@link ComplexPolynomial} matrices.
	 */
	public ComplexPolynomial[][] add(ComplexPolynomial[][] p, ComplexPolynomial[][] q) {
		return matrixAdd.add(p, q);
	}
	
	/**
	 * Multiply two {@link ComplexPolynomial} matrices.
	 */
	public ComplexPolynomial[][] mul(ComplexPolynomial[][] p1, ComplexPolynomial[][] p2) {
		return matrixMultiply.multiply(p1, p2);
	}
	
	public void release() {
		add.release();
		multiply.release();
		matrixAdd.release();
		matrixMultiply.release();
	}
}
