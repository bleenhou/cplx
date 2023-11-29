package eth.bram.cplx;

public interface GPUProgram {

	GPUKernel buildKernel(String kernelName);

}
