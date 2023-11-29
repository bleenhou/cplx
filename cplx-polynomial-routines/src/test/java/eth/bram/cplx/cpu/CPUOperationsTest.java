package eth.bram.cplx.cpu;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import eth.bram.cplx.ComplexPolynomial;

/**
 * Basic testing of the CPU functions
 */
public class CPUOperationsTest {

	@Test
	public void basicAdd() {
		final ComplexPolynomial p1 = new ComplexPolynomial(new float[] { 2, 3, 5 }, new float[] { 7, 11, 13 });
		final ComplexPolynomial p2 = new ComplexPolynomial(new float[] { 17, 19, 23 }, new float[] {29, 31, 37});
		final ComplexPolynomial add = CPUOperations.add(p1, p2);
		final ComplexPolynomial expected = new ComplexPolynomial(new float[] {19, 22, 28}, new float[] {36, 42, 50});
		check(add, expected);
	}

	@Test
	public void basicMultiply() {
		final ComplexPolynomial p1 = new ComplexPolynomial(new float[] { 2, 3, 5 }, new float[] { 7, 11, 13 });
		final ComplexPolynomial p2 = new ComplexPolynomial(new float[] { 17, 19, 23 }, new float[] {29, 31, 37});
		final ComplexPolynomial mul = CPUOperations.mul(p1, p2);
		final ComplexPolynomial expected = new ComplexPolynomial(new float[] {-169, -447, -789}, new float[] {177, 469, 903});
		check(mul, expected);
	}
	
	private static void check(ComplexPolynomial add, ComplexPolynomial expected) {
		Assertions.assertArrayEquals(expected.re().array(), add.re().array());
		Assertions.assertArrayEquals(expected.im().array(), add.im().array());
	}
}
