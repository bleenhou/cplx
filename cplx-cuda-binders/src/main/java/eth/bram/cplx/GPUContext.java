package eth.bram.cplx;

import java.nio.ByteBuffer;
import java.util.List;

public interface GPUContext {

	static GPUContext create(int deviceId) {
		return  new CudaContext(deviceId);
	}
	
	static List<Integer> listDevices() {
		return CudaContext.listDevices();
	}
	
	GPUProgram createProgram(List<String> sources);

	GPUBuffer createBuffer(ByteBuffer b);
	
	GPUDeviceBuffer createDeviceBuffer(long size);

	void release();
}
