package eth.bram.cplx;

import static eth.bram.cplx.CudaContext.MUTEX;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFreeAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoDAsync;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;

import java.nio.ByteBuffer;

import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;

class CudaDeviceBuffer implements GPUDeviceBuffer {

	private final CUstream stream;
	private final CUdeviceptr ptr;
	
	CudaDeviceBuffer(CUcontext context, CUstream stream, long size) {
		this.stream = stream;
	    this.ptr = new CUdeviceptr();
	    synchronized (MUTEX) {
	    	JCudaDriver.cuCtxSetCurrent(context);
		    cuMemAlloc(ptr, size);
		}
	}
	
	@Override
	public GPUPointer pointer() {
		return new CudaPointer(Pointer.to(ptr));
	}
	
	@Override
	public void write(long offset, ByteBuffer buffer) {
		cuMemcpyHtoDAsync(ptr.withByteOffset(offset), Pointer.to(buffer.array()), buffer.capacity(), stream);
		cuStreamSynchronize(stream);
	}

	void release() {
	    cuMemFreeAsync(ptr, stream);
	    cuStreamSynchronize(stream);
	}

	@Override
	public ByteBuffer read(ByteBuffer data) {
		JCudaDriver.cuMemcpyDtoHAsync(Pointer.to(data.array()), ptr, data.capacity(), stream);
		cuStreamSynchronize(stream);
		return data;
	}
}
