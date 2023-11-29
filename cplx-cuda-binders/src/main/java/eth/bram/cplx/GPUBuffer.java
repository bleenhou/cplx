package eth.bram.cplx;

import java.nio.ByteBuffer;

public interface GPUBuffer {

	GPUPointer pointer();

	ByteBuffer buffer();

	void read();

	void write();

}
