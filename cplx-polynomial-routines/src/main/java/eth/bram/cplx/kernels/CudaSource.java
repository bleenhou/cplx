package eth.bram.cplx.kernels;

public class CudaSource {
	
	public static String getSource(String file) {
		try {
			return new String(CudaSource.class.getResourceAsStream(file).readAllBytes());
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
	
}