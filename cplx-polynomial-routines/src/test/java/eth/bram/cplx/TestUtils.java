package eth.bram.cplx;

import java.nio.ByteBuffer;
import java.util.Arrays;

import org.junit.jupiter.api.Assertions;

class TestUtils {

	/**
	 * Compares a and b, throw an exception if not equal
	 */
	static void compare (ComplexPolynomial a, ComplexPolynomial b) {
		Assertions.assertArrayEquals(toFloatArray(a.re()), toFloatArray(b.re()), 0.1f);
		Assertions.assertArrayEquals(toFloatArray(a.im()), toFloatArray(b.im()), 0.1f);
	}
	
	static void print(ComplexPolynomial a) {
		System.out.println(Arrays.toString(toFloatArray(a.re())));
	}
	
	/**
	 * Convert to float array
	 */
	static final float [] toFloatArray(ByteBuffer b) {
		final float [] res = new float [b.capacity() / Float.BYTES];
		for (int i = 0 ; i < res.length ; ++i) {
			res[i] = b.getFloat(i * Float.BYTES);
		}
		return res;
	}
}
