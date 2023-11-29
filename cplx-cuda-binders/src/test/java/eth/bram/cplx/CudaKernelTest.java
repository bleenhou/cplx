package eth.bram.cplx;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

public class CudaKernelTest {

	@Test
	public void createGPUBuffer() {
		final GPUContext c = new CudaContext(0);
		for (int i = 0 ; i < 10 ; ++i) {
			final GPUProgram p = c.createProgram(Arrays.asList("extern \"C\" __global__ void name() {}"));
			final GPUKernel k = p.buildKernel("name");
			k.run(1, 1);
		}
	}
}
