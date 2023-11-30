package eth.bram.cplx.cpu;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * A polynomial of which coefficients are complex numbers.
 * (a0 + b0 * i) + (a1 + b1 *i) * X + ... + (an + bn * i) * X^n
 */
public class ComplexPolynomial {
	
	/**
	 * coefficients for each X^n in ascending order.
	 */
	private final ByteBuffer data;
	
	/**
	 * Builds a complex polynomial with the provided coefficients.
	 */
	public ComplexPolynomial(float [] real, float [] imaginary){
		if (imaginary.length != real.length) {
			throw new IllegalArgumentException("Imaginary and real part must have the same size.");
		}
		this.data = createByteBuffer(real, imaginary);
	}
	
	/**
	 * Builds a complex polynomial with the provided coefficients.
	 */
	public ComplexPolynomial(ByteBuffer data){
		this.data = data;
	}
	
	/**
	 * Creates a buffer from a float array with correct alignment
	 */
	private static ByteBuffer createByteBuffer(float [] re, float [] im) {
		if (re.length != im.length) {
			throw new IllegalArgumentException("Real and immaginary part must have the same length");
		}
		final ByteBuffer b = ByteBuffer.allocate((re.length + im.length) * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0 ; i < re.length ; ++i) {
			b.putFloat(re[i]);
			b.putFloat(im[i]);
		}
		return b.rewind();
	}
	
	/**
	 * Getter for the real part of the coefficients
	 */
	public ByteBuffer data() {
		return this.data;
	}
	
	/**
	 * Size of the polynomial.
	 */
	public int size(){
		return data.capacity() / (Float.BYTES * 2);
	}

	ByteBuffer re() {
		final ByteBuffer b = ByteBuffer.allocate(size() * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0 ; i < size() ; ++i) {
			b.putFloat(data.getFloat(2 * i * Float.BYTES));
		}
		return b.rewind();
	}
	
	ByteBuffer im() {
		final ByteBuffer b = ByteBuffer.allocate(size() * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0 ; i < size() ; ++i) {
			b.putFloat(data.getFloat((2 * i + 1) * Float.BYTES));
		}
		return b.rewind();
	}
}
