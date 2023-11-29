package eth.bram.cplx;

import java.util.Random;

/**
 * Generators for polynomials and matrices of polynomials.
 * Generation is deterministic with a fixed RNG seed.
 */
public class ComplexPolynomialGenarator {

	private static final Random r = new Random(1234);

	/**
	 * Creates a random polynomial of the requested size.
	 * @param size size of the polynomial.
	 * @return a random polynomial of the requested size.
	 */
	public static ComplexPolynomial create(int size) {
		final float[] real = new float[size];
		final float[] immaginary = new float[size];
		for (int i = 0; i < size; ++i) {
			real[i] = r.nextFloat(1);
			immaginary[i] = r.nextFloat(1);
		}
		return new ComplexPolynomial(real, immaginary);
	}

	/**
	 * Creates a height * width random polynomial matrix of the requested size.
	 * @param size size of the polynomial.
	 * @param height height of the matrix.
	 * @param width width of the matrix.
	 * @return a height * width random polynomial matrix of the requested size.
	 */
	public static ComplexPolynomial [][] createMatrix(int size, int height, int width) {
		final ComplexPolynomial [][] matrix = new ComplexPolynomial[height][width];
		for (int i = 0 ; i < height ; ++i) {
			for (int j = 0 ; j < width ; ++j) {
				matrix[i][j] = create(size);
			}
		}
		return matrix;
	}
}
