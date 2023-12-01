package eth.bram.cplx;

import org.junit.jupiter.api.Test;

import eth.bram.cplx.cpu.CPUOperations;
import eth.bram.cplx.cpu.ComplexPolynomial;

/**
 * Performs testing of Addition and multiplication of polynomials on GPU.
 * Tests operations on polynomials of size 2^n (n ranging from 1 to 15)
 */
public class GPUBaseOperationsTest {

	@Test
	public void testAdd() {
		for (int size = 1, i = 0 ; i < 15 ; size *= 2, i++) {
			System.out.println("[CPU] Addition of polynomials of size " + size);
			final ComplexPolynomial p1 = ComplexPolynomialGenarator.create(size);
			final ComplexPolynomial p2 = ComplexPolynomialGenarator.create(size);
			final ComplexPolynomial cpuSum = CPUOperations.add(p1, p2);
	
			System.out.println("[GPU] Addition of polynomials of size " + size);
			final ComplexPolynomialOperations kernels = new ComplexPolynomialOperations(size);
			final ComplexPolynomial gpuSum = kernels.add(p1, p2);
			kernels.release();
			TestUtils.compare(cpuSum, gpuSum);
		}
	}
	
	@Test
	public void testMultiply() {
		for (int size = 2, i = 0 ; i < 14 ; size *= 2, i++) {
			testMultiply(size);
		}
	}

	private void testMultiply(int size) {
		System.out.println("[CPU] Multiplication of polynomials of size " + size);
		final ComplexPolynomial p1 = ComplexPolynomialGenarator.create(size);
		final ComplexPolynomial p2 = ComplexPolynomialGenarator.create(size);
		final ComplexPolynomial cpuMul = CPUOperations.mul(p1, p2);
		
		System.out.println("[GPU] Multiplication of polynomials of size " + size);
		final ComplexPolynomialOperations kernels = new ComplexPolynomialOperations(size);
		final ComplexPolynomial gpuMul = kernels.multiply(p1, p2);
		kernels.release();
		TestUtils.compare(cpuMul, gpuMul);
	}
}
