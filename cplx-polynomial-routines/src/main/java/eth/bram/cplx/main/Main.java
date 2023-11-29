package eth.bram.cplx.main;

import eth.bram.cplx.ComplexPolynomial;
import eth.bram.cplx.ComplexPolynomialGenarator;
import eth.bram.cplx.ComplexPolynomialOperations;

public class Main {

	public static void main(String[] args) {
		
		if (args.length < 1) {
			printUsage();
			return;
		}
		
		if ("ADD".equals(args[0])) {
			add(1 << Integer.parseInt(args[1]));
			return;
		}
		
		if ("MUL".equals(args[0])) {
			mulFFT(1 << Integer.parseInt(args[1]));
			return;
		}
		
		if ("ADD_MAT".equals(args[0])) {
			addMatrix(1 << Integer.parseInt(args[1]), Integer.parseInt(args[2]), Integer.parseInt(args[3]));
			return;
		}
		
		if ("MUL_MAT".equals(args[0])) {
			mulMatrix(1 << Integer.parseInt(args[1]), Integer.parseInt(args[2]), Integer.parseInt(args[3]));
			return;
		}
		
		if ("BENCHMARK".equals(args[0])) {
			benchmark();
			return;
		}
		
		printUsage();
	}

	private static void benchmark() {
		System.out.println("Benchmarking polynomial Addition...");
		try {
			for (int i = 10 ; i < 26 ; ++i) {
				add(1 << i);
			}
		} catch (OutOfMemoryError e) {
			System.out.println("Out of cpu memory, consider increasing -Xmx jvm parameter");
		}
		
		System.out.println();
		System.out.println("Benchmarking polynomial multiplication...");
		try {
			for (int i = 10 ; i < 26 ; ++i) {
				mulFFT(1 << i);
			}
		} catch (OutOfMemoryError e) {
			System.out.println("Out of cpu memory, consider increasing -Xmx jvm parameter");
		}
		
		System.out.println();
		System.out.println("Benchmarking matrix addition...");
		try {
			for (int i = 0, size = 64 ; i < 6 ; ++i, size *= 2) {
				addMatrix(1 << 8, size, size);
			}
		} catch (OutOfMemoryError e) {
			System.out.println("Out of cpu memory, consider increasing -Xmx jvm parameter");
		}
		
		System.out.println();
		System.out.println("Benchmarking matrix multiplication...");
		try {
			for (int i = 0, size = 16; i < 7; ++i, size *= 2) {
				mulMatrix(1 << 10, size, size);
			}
		} catch (OutOfMemoryError e) {
			System.out.println("Out of cpu memory, consider increasing -Xmx jvm parameter");
		}
	}
	
	private static void mulMatrix(int size, int height, int width) {
		System.out.print(String.format("Creating 2 * %-12s size %-4s polynomials     ", String.format("[%d][%d]", height, width), String.valueOf(size)));
		final ComplexPolynomial [][] p1 = ComplexPolynomialGenarator.createMatrix(size, height, width);
		final ComplexPolynomial [][] p2 = ComplexPolynomialGenarator.createMatrix(size, width, height);
		final ComplexPolynomialOperations gpuKernels = new ComplexPolynomialOperations(size);
		System.out.print("Multiplying... ");
		final long then = System.currentTimeMillis();
		gpuKernels.mul(p1, p2);
		System.out.println("Done in " + (System.currentTimeMillis() - then) + " ms");
		gpuKernels.release();
	}
	
	private static void addMatrix(int size, int height, int width) {
		System.out.print(String.format("Creating 2 * %-12s size %-4s polynomials     ", String.format("[%d][%d]", height, width), String.valueOf(size)));
		final ComplexPolynomial [][] p1 = ComplexPolynomialGenarator.createMatrix(size, height, width);
		final ComplexPolynomial [][] p2 = ComplexPolynomialGenarator.createMatrix(size, height, width);
		final ComplexPolynomialOperations gpuKernels = new ComplexPolynomialOperations(size);
		System.out.print("Adding... ");
		final long then = System.currentTimeMillis();
		gpuKernels.add(p1, p2);
		System.out.println("Done in " + (System.currentTimeMillis() - then) + " ms");
		gpuKernels.release();
	}

	private static void mulFFT(final int size) {
		System.out.print(String.format("Creating two polynomials of size %-12s", String.valueOf(size)));
		final ComplexPolynomial p1 = ComplexPolynomialGenarator.create(size);
		final ComplexPolynomial p2 = ComplexPolynomialGenarator.create(size);
		final ComplexPolynomialOperations gpuKernels = new ComplexPolynomialOperations(size);
		System.out.print("Multiplying... ");
		final long then = System.currentTimeMillis();
		gpuKernels.multiply(p1, p2);
		System.out.println("Done in " + (System.currentTimeMillis() - then) + " ms");
		gpuKernels.release();
	}

	private static void add(final int size) {
		System.out.print(String.format("Creating two polynomials of size %-12s", String.valueOf(size)));
		final ComplexPolynomial p1 = ComplexPolynomialGenarator.create(size);
		final ComplexPolynomial p2 = ComplexPolynomialGenarator.create(size);
		final ComplexPolynomialOperations gpuKernels = new ComplexPolynomialOperations(size);
		System.out.print("Adding...");
		final long then = System.currentTimeMillis();
		gpuKernels.add(p1, p2);
		System.out.println(" Done in " + (System.currentTimeMillis() - then) + " ms");
		gpuKernels.release();
	}

	private static void printUsage() {
		System.out.println("Usage : java -Xmx16g -jar cplx-jar-with-dependencies.jar [mode] [n] [height] [width]");
		System.out.println();
		System.out.println("\t Parameters :");
		System.out.println("\t\t Xmx    	: Max CPU heap size.");
		System.out.println("\t\t mode   	: Computation to run (see below).");
		System.out.println("\t\t n      	: Size of the polynomial.");
		System.out.println("\t\t height 	: Height of the matrix for matrix mode.");
		System.out.println("\t\t width  	: Width of the matrix for matrix mode.");
		
		System.out.println();
		System.out.println("\t Modes :");
		System.out.println("\t\t ADD 		: Adds two polynomials of size 2^n.");
		System.out.println("\t\t MUL 		: Multiplies two polynomials of size 2^n.");
		System.out.println("\t\t ADD_MAT 	: Adds two matrices of polynoials of size 2^n.");
		System.out.println("\t\t MUL_MAT 	: Adds two matrices of polynoials of size 2^n.");
		System.out.println("\t\t BENCHMARK 	: Benchmarks all modes with varying parameters.");
		
		System.out.println();
		System.out.println("\t Examples :");
		System.out.println("\t\t java -Xmx16g -jar cplx-jar-with-dependencies.jar ADD 15");
		System.out.println("\t\t java -Xmx16g -jar cplx-jar-with-dependencies.jar MUL 23");
		System.out.println("\t\t java -Xmx16g -jar cplx-jar-with-dependencies.jar ADD_MAT 15 64 92");
		System.out.println("\t\t java -Xmx16g -jar cplx-jar-with-dependencies.jar MUL_MAT 10 64 64");
		System.out.println("\t\t java -Xmx16g -jar cplx-jar-with-dependencies.jar BENCHMARK");
	}

}
