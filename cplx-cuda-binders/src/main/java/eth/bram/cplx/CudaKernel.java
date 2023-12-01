package eth.bram.cplx;

import static eth.bram.cplx.CudaContext.MUTEX;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;

import java.util.Map;
import java.util.TreeMap;

import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;

class CudaKernel implements GPUKernel {

	private final CUstream stream;
	private final CUmodule module;
	private final CUfunction function;
	private final Map<Integer, Pointer> parameters = new TreeMap<>();

	CudaKernel(CUcontext context, CUstream stream, String ptx, String name) {
		this.module = new CUmodule();
		this.function = new CUfunction();
	    synchronized (MUTEX) {
	    	JCudaDriver.cuCtxSetCurrent(context);
	    	cuModuleLoadData(module, ptx);
			cuModuleGetFunction(function, module, name);
	    }
		this.stream = stream;
	}

	@Override
	public void setArg(int pos, GPUPointer p) {
		parameters.put(pos, ((CudaPointer) p).pointer());
	}

	@Override
	public void setArg(int pos, int value) {
		parameters.put(pos, Pointer.to(new int[] { value }));
	}

	@Override
	public void setArg(int pos, long value) {
		parameters.put(pos, Pointer.to(new long[] { value }));
	}

	@Override
	public void run(long size, Integer localSize) {
		if(localSize != null && size % localSize != 0) {
			run(size);
			return;
		}
		final Pointer p = Pointer.to(parameters.values().toArray(new Pointer [parameters.size()]));
		final int localSz = (int) (localSize == null ? GPUKernel.gcd(size, 32) : localSize);
		cuLaunchKernel(function, (int) (size / localSz), 1, 1, localSz, 1, 1, 0, stream, p, null);
		JCudaDriver.cuStreamSynchronize(stream);
	}
	
	@Override
	public void run(long size) {
		run(size, null);
	}

	void release() {
	    JCudaDriver.cuModuleUnload(module);
	}
}
