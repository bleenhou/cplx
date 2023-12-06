package eth.bram.cplx.kernels;

public class CudaSource {
	
	public static String getSource() {
		return getSource(32);
	}
	
	public static String getSource(int localSizeMatrixMultiply) {
		try {
			return new String(CudaSource.class.getResourceAsStream("kernels.cu").readAllBytes()).replaceAll("<LOCAL_SIZE_MATRIX_MULTIPLY>", String.valueOf(localSizeMatrixMultiply));
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
	
}