package eth.bram.cplx;

import java.nio.ByteBuffer;

public interface GPUDeviceBuffer {

	GPUPointer pointer();

	void write(long offset, ByteBuffer data);

	ByteBuffer read(ByteBuffer data);

}
