package eth.bram.cplx;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import eth.bram.cplx.cpu.ComplexPolynomial;

/**
 * Generators for polynomials and matrices of polynomials.
 * Generation is deterministic with a fixed RNG seed.
 */
public class ComplexPolynomialGenarator {

	private static final Random R = new Random(1234);

	/**
	 * Creates a random polynomial of the requested size.
	 * @param size size of the polynomial.
	 * @return a random polynomial of the requested size.
	 */
	public static ComplexPolynomial create(int size) {
		return create(size, R);
	}
	
	/**
	 * Creates a random polynomial of the requested size.
	 * @param size size of the polynomial.
	 * @return a random polynomial of the requested size.
	 */
	private static ComplexPolynomial create(int size, Random r) {
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
		final ExecutorService exc = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
		final List<Future<List<ComplexPolynomial>>> futures = new ArrayList<>();
		for (int i = 0 ; i < height ; ++i) {
			final Random r = new Random(R.nextLong());
			futures.add(exc.submit(() -> {
				final List<ComplexPolynomial> line = new ArrayList<>();
				for (int j = 0 ; j < width ; ++j) {
					line.add(create(size, r));
				}
				return line;
			}));
		}
		for (int i = 0 ; i < height ; ++i) {
			final List<ComplexPolynomial> line = ensureFinished(futures.get(i));
			for (int j = 0 ; j < width ; ++j) {
				matrix[i][j] = line.get(j);
			}
		}
		return matrix;
	}
	
	/**
	 * Ensures the thread is finished and gets its return value
	 */
	private static <V> V ensureFinished(Future<V> f) {
		try {
			return f.get();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
}
