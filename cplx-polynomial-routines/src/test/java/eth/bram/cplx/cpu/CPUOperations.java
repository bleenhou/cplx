package eth.bram.cplx.cpu;

import java.nio.ByteBuffer;

import eth.bram.cplx.ComplexPolynomial;

/**
 * CPU implementation of the operations over the complex polynomial ring.
 */
public class CPUOperations {

	/**
	 * Add two {@link ComplexPolynomial}.
	 */
	public static ComplexPolynomial add(ComplexPolynomial p1, ComplexPolynomial p2) {
		final float [] real = add(p1.re(), p2.re());
		final float [] imaginary = add(p1.im(), p2.im());
		return new ComplexPolynomial(real, imaginary);
	}
	
	/**
	 * Multiply two {@link ComplexPolynomial}.
	 */
	public static ComplexPolynomial mul(ComplexPolynomial p1, ComplexPolynomial p2) {
		final float [] a = mul(p1.im(), p2.im());
		final float [] b = mul(p1.re(), p2.im());
		final float [] c = mul(p1.im(), p2.re());
		final float [] d = mul(p1.re(), p2.re());
		return new ComplexPolynomial(sub(d, a), add(b, c));
	}
	
	/**
	 * Add two {@link ComplexPolynomial} matrices.
	 */
	public static ComplexPolynomial[][] add(ComplexPolynomial[][] p, ComplexPolynomial[][] q) {
		final ComplexPolynomial[][] s = new ComplexPolynomial [p.length][p[0].length];
		for (int i = 0 ; i < s.length ; ++i) {
			for (int j = 0 ; j < s[i].length ; ++j) {
				s[i][j] = add(p[i][j], q[i][j]);
			}
		}
		return s;
	}
	
	/**
	 * Multiply two {@link ComplexPolynomial} matrices.
	 */
	public static ComplexPolynomial[][] mul(ComplexPolynomial[][] p, ComplexPolynomial[][] q) {
		final ComplexPolynomial[][] s = new ComplexPolynomial [p.length][p.length];
		final int size = p[0][0].size();
		for (int i = 0 ; i < p.length ; ++i) {
			for (int j = 0 ; j < p.length ; ++j) {
				ComplexPolynomial sum = new ComplexPolynomial(new float[size], new float[size]);
				for (int k = 0 ; k < q.length ; ++k) {
					sum = add(sum, mul(p[i][k], q[k][j]));
				}
				s[i][j] = sum;
			}
		}
		return s;
	}
	
	/**
	 * Substract two polynomials.
	 */
	private static float [] sub (float [] lhs, float [] rhs) {
		final float [] res = new float [lhs.length];
		for (int i = 0 ; i < res.length ; ++i) {
			res[i] += lhs[i] - rhs[i];
		}
		return res;
	}
	
	/**
	 * Add two polynomials
	 */
	private static float [] add (float [] lhs, float [] rhs) {
		final float [] res = new float [lhs.length];
		for (int i = 0 ; i < res.length ; ++i) {
			res[i] += lhs[i] + rhs[i];
		}
		return res;
	}
	
	/**
	 * Add two polynomials
	 */
	private static float [] add (ByteBuffer lhs, ByteBuffer rhs) {
		final float [] res = new float [lhs.capacity() / Float.BYTES];
		for (int i = 0 ; i < res.length ; ++i) {
			res[i] += lhs.getFloat(i * Float.BYTES) + rhs.getFloat(i * Float.BYTES);
		}
		return res;
	}
	
	/**
	 * Multiply two polynomials mod(lhs.length)
	 */
	private static float [] mul (ByteBuffer lhs, ByteBuffer rhs) {
		final float [] res = new float [lhs.capacity() / Float.BYTES];
		for (int i = 0 ; i < res.length ; ++i) {
			for (int j = 0 ; j < lhs.capacity() / Float.BYTES - i ; ++j) {
				res[i + j] += lhs.getFloat(i * Float.BYTES) * rhs.getFloat(j * Float.BYTES);
			}
		}
		return res;
	}
}
