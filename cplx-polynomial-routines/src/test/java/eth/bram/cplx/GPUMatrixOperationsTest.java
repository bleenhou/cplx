package eth.bram.cplx;

import org.junit.jupiter.api.Test;

import eth.bram.cplx.cpu.CPUOperations;

/**
 * Performs testing of Addition and multiplication of polynomials on GPU.
 */
public class GPUMatrixOperationsTest {

	/**
	 * Tests matrix addition from sizes 16 to 20 for polynomials of degree 2^2 to 2^10
	 * (Testing n * (n+1) matrices to make sure square matrices are not a requirement)
	 */
	@Test
	public void matrixGPUAdd() {
		for (int size = 2, i = 0 ; i < 10 ; size *= 2, i++) {
			final ComplexPolynomialOperations kernels = new ComplexPolynomialOperations(size);
			for (int matrixSize = 1 ; matrixSize < 16 ; matrixSize++) {
				testAdd(kernels, size, matrixSize, matrixSize +1);
			}
			kernels.release();
		}
	}
	
	/**
	 * Tests matrix multiplication from sizes 1 to 20 for polynomials of degree 2^2 to 2^8
	 * (Testing n * (n+1) matrices to make sure square matrices are not a requirement)
	 */
	@Test
	public void matrixGPUMultiply() {
		for (int size = 2, i = 0 ; i < 8 ; size *= 2, i++) {
			final ComplexPolynomialOperations kernels = new ComplexPolynomialOperations(size);
			for (int matrixSize = 1 ; matrixSize < 20 ; matrixSize++) {
				testMul(kernels, size, matrixSize, matrixSize + 1);
			}
			kernels.release();
		}
	}
	
	/**
	 * Tests matrix addition with slicing
	 */
	@Test
	public void matrixGPUAddSlicing() {
		final int polynomialSize = 4;
		final ComplexPolynomialOperations kernels = new ComplexPolynomialOperations(polynomialSize, 2048, 2048);
		testAdd(kernels, polynomialSize, 23, 17);
		kernels.release();
	}
	
	/**
	 * Tests matrix multiplication with slicing
	 */
	@Test
	public void matrixGPUMultiplySlicing() {
		final int polynomialSize = 4;
		final ComplexPolynomialOperations kernels = new ComplexPolynomialOperations(polynomialSize, 1024, 1024);
		testMul(kernels, polynomialSize, 20, 20);
		kernels.release();
	}
	
	/**
	 * Tests matrix multiplication with slicing
	 */
	@Test
	public void matrixGPUMultiplyImperfectSlicing() {
		final int polynomialSize = 4;
		final ComplexPolynomialOperations kernels = new ComplexPolynomialOperations(polynomialSize, 4096, 2500);
		testMul(kernels, polynomialSize, 20, 20);
		kernels.release();
	}

	/**
	 * Compares a matrix addition GPU to CPU reference. 
	 * @param size size of the polynomial
	 * @param height height of matrices
	 * @param width width of matrices
	 */
	private void testAdd(ComplexPolynomialOperations kernels, int size, int height, int width) {
		final ComplexPolynomial[][] p1 = ComplexPolynomialGenarator.createMatrix(size, height, width);
		final ComplexPolynomial[][] p2 = ComplexPolynomialGenarator.createMatrix(size, height, width);
		
		System.out.println("[CPU] Addition of matrices " + height + " * " + width + " of polynomials of size " + size);
		final ComplexPolynomial[][] cpuSum = CPUOperations.add(p1, p2);

		System.out.println("[GPU] Addition of matrices " + height + " * " + width + " of polynomials of size " + size);
		final ComplexPolynomial[][] gpuSum = kernels.add(p1, p2);

		for (int j = 0 ; j < cpuSum.length ; ++j) {
			for (int k = 0 ; k < cpuSum[j].length ; ++k) {
				TestUtils.compare(cpuSum[j][k], gpuSum[j][k]);
			}
		}
	}
	
	/**
	 * Compares a matrix multiplication GPU to CPU reference.
	 * @param size size of the polynomial
	 * @param height height of matrix A 
	 * @param width width of matrix B
	 */
	private void testMul(ComplexPolynomialOperations kernels, int size, int height, int width) {
		final ComplexPolynomial[][] p1 = ComplexPolynomialGenarator.createMatrix(size, height, width);
		final ComplexPolynomial[][] p2 = ComplexPolynomialGenarator.createMatrix(size, width, height);
		
		System.out.println("[CPU] Multiplication of matrices " + height + " * " + width + " of polynomials of size " + size);
		final ComplexPolynomial[][] cpuMul = CPUOperations.mul(p1, p2);

		System.out.println("[GPU] Multiplication of matrices " + height + " * " + width + " of polynomials of size " + size);
		final ComplexPolynomial[][] gpuMul = kernels.mul(p1, p2);
		
		for (int j = 0 ; j < cpuMul.length ; ++j) {
			for (int k = 0 ; k < cpuMul[j].length ; ++k) {
				TestUtils.compare(cpuMul[j][k], gpuMul[j][k]);
			}
		}
	}
}
