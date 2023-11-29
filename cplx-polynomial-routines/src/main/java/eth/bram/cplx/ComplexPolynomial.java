package eth.bram.cplx;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * A polynomial of which coefficients are complex numbers.
 * (a0 + b0 * i) + (a1 + b1 *i) * X + ... + (an + bn * i) * X^n
 */
public class ComplexPolynomial {
	
	/**
	 * Real part of the coefficients for each X^n in ascending order.
	 */
	private final ByteBuffer re;
	
	/**
	 * Imaginary part of the coefficients for each X^n in ascending order.
	 */
	private final ByteBuffer im;
	
	/**
	 * Builds a complex polynomial with the provided coefficients.
	 */
	public ComplexPolynomial(float [] real, float [] imaginary){
		if (imaginary.length != real.length) {
			throw new IllegalArgumentException("Imaginary and real part must have the same size.");
		}
		this.re = createByteBuffer(real);
		this.im = createByteBuffer(imaginary);
	}
	
	/**
	 * Builds a complex polynomial with the provided coefficients.
	 */
	public ComplexPolynomial(ByteBuffer real, ByteBuffer imaginary){
		if (imaginary.capacity() != real.capacity()) {
			throw new IllegalArgumentException("Imaginary and real part must have the same size.");
		}
		this.re = real;
		this.im = imaginary;
	}
	
	/**
	 * Creates a buffer from a float array with correct alignment
	 */
	private static ByteBuffer createByteBuffer(float [] buffer) {
		final ByteBuffer b = ByteBuffer.allocate(buffer.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0 ; i < buffer.length ; ++i) {
			b.putFloat(buffer[i]);
		}
		return b.rewind();
	}
	
	/**
	 * Getter for the real part of the coefficients
	 */
	public ByteBuffer re() {
		return this.re;
	}
	
	/**
	 * Getter for the imaginary part of the coefficients
	 */
	public ByteBuffer im() {
		return this.im;
	}
	
	/**
	 * Size of the polynomial.
	 */
	public int size(){
		return re.capacity() / Float.BYTES;
	}
}
