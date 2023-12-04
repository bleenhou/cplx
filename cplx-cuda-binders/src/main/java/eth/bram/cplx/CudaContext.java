package eth.bram.cplx;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSetCurrent;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuStreamCreate;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.nvrtc.JNvrtc;

class CudaContext implements GPUContext {
	
	static final Object MUTEX = new Object();
	private static int deviceCount = -1;
	
	private final CUcontext context;
	private final CUdevice device;
	private final CUstream stream;
	private final List<CudaBuffer> buffers = new CopyOnWriteArrayList<>();
	private final List<CudaDeviceBuffer> deviceBuffers = new CopyOnWriteArrayList<>();
	private final List<CudaProgram> programs = new ArrayList<>();
	
	CudaContext(int deviceId) {
		lazyInit();
		this.device = new CUdevice();
		this.context = new CUcontext();
		this.stream = new CUstream();
		cuDeviceGet(device, deviceId);
		cuCtxCreate(context, 0, device);
		cuCtxSetCurrent(context);
		cuStreamCreate(stream, 0);
	}
	
	private synchronized static final void lazyInit() {
		if (deviceCount != -1) {
			return;
		}
		JCudaDriver.setExceptionsEnabled(true);
		JNvrtc.setExceptionsEnabled(true);
		JCudaDriver.cuInit(0);
		
		// get device count
        int deviceCountArray[] = { 0 };
        JCudaDriver.cuDeviceGetCount(deviceCountArray);
        deviceCount = deviceCountArray[0];
	}
	
	@Override
	public GPUProgram createProgram(List<String> sources) {
		final CudaProgram program = new CudaProgram(context, stream, sources);
		programs.add(program);
		return program;
	}
	
	@Override
	public GPUBuffer createBuffer(ByteBuffer b) {
		final CudaBuffer buffer = new CudaBuffer(context, stream, b.order(ByteOrder.LITTLE_ENDIAN));
		buffers.add(buffer);
		return buffer;
    }
	
	@Override
	public GPUDeviceBuffer createDeviceBuffer(long size) {
		final CudaDeviceBuffer buffer = new CudaDeviceBuffer(context, stream, size);
		deviceBuffers.add(buffer);
		return buffer;
    }
	
	@Override
	public void release() {
		programs.forEach(p -> p.release());
		buffers.forEach(b -> b.release());
		deviceBuffers.forEach(b -> b.release());
		JCudaDriver.cuStreamDestroy(stream);
		JCudaDriver.cuCtxDestroy(context);
	}
	
	static List<Integer> listDevices(){
		lazyInit();
		return IntStream.range(0, deviceCount).mapToObj(x -> x).collect(Collectors.toList());
	}
}
