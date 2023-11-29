package eth.bram.cplx;

public interface GPUKernel {

	void setArg(int pos, GPUPointer gpuPointer);

	void setArg(int pos, int value);

	void setArg(int pos, long value);

	void run(long size, Integer localSize);
	
	void run(long size);
	
	public static long gcd(long a, long b) {   
		return (b == 0) ? a : gcd(b, a % b);   
	}  

}
