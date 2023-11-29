package eth.bram.cplx;

import static eth.bram.cplx.CudaContext.MUTEX;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFreeAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoHAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoDAsync;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;

import java.nio.ByteBuffer;

import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;

class CudaBuffer implements GPUBuffer {

	private final CUstream stream;
	private final ByteBuffer buffer;
	private final CUdeviceptr ptr;
	
	CudaBuffer(CUcontext context, CUstream stream, ByteBuffer buffer) {
		this.stream = stream;
		this.buffer = buffer;
	    this.ptr = new CUdeviceptr();
	    synchronized (MUTEX) {
	    	JCudaDriver.cuCtxSetCurrent(context);
		    cuMemAlloc(ptr, buffer.capacity());
		}
	    write();
	}
	
	@Override
	public ByteBuffer buffer() {
		return buffer;
	}
	
	@Override
	public GPUPointer pointer() {
		return new CudaPointer(Pointer.to(ptr));
	}
	
	@Override
	public void write() {
		cuMemcpyHtoDAsync(ptr, Pointer.to(buffer.array()), buffer.capacity(), stream);
		cuStreamSynchronize(stream);
	}

	@Override
	public void read() {
		cuMemcpyDtoHAsync(Pointer.to(buffer.array()), ptr, buffer.capacity(), stream);
		cuStreamSynchronize(stream);
	}
	
	void release() {
		// TODO : 
		// Consider using sync free if it works with multiple devices
		// Seems like older GPUs do not support async free well (980 cards)
//		JCudaDriver.cuMemFree(ptr);
	    cuMemFreeAsync(ptr, stream);
	    cuStreamSynchronize(stream);
	}
}
