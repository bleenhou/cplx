package eth.bram.cplx;

import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import jcuda.CudaException;
import jcuda.driver.CUcontext;
import jcuda.driver.CUstream;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;

class CudaProgram implements GPUProgram {
	
	private static final String[] COMPILE_OPTIONS = new String [] {};
	private static final Map<String, String> PTX_CACHE = new ConcurrentHashMap<>();
	private static final String MACROS = readMacros();
	
	private final String ptx;
	private final CUstream stream;
	private final CUcontext context;
	private final List<CudaKernel> kernels = new ArrayList<>();
	
	CudaProgram(CUcontext context, CUstream stream, List<String> sources){
		final String source = MACROS + "\n" + sources.stream().collect(Collectors.joining("\n"));
		final String sourceHash = getHash(source);
        this.ptx = PTX_CACHE.computeIfAbsent(sourceHash, s -> readOrBuildPtx(s, source));
        this.stream = stream;
        this.context = context;
	}

	private String getHash(final String source) {
		try {
			final String data = Arrays.toString(COMPILE_OPTIONS) + source;
			return bytesToHex(MessageDigest.getInstance("SHA-256").digest(data.getBytes(StandardCharsets.UTF_8)));
		} catch (NoSuchAlgorithmException e) {
			throw new IllegalStateException(e);
		}
	}
	
	private static String readOrBuildPtx(String sourceHash, String source) {
		
		// Try to read from disk cache
		final File cache = getCacheFile(sourceHash);
		if (cache.exists()) {
			return safeRead(cache);
		}
		
		final nvrtcProgram program = new nvrtcProgram();
		nvrtcCreateProgram(program, source, null, 0, null, null);
        try {
        	nvrtcCompileProgram(program, COMPILE_OPTIONS.length, COMPILE_OPTIONS);
        } catch (CudaException e) {
            final String[] error = new String[1];
        	JNvrtc.nvrtcGetProgramLog(program, error);
        	int lineIdx = 1;
        	for (String line : source.split("\n")) {
        		System.err.println("[" + lineIdx++ + "]\t" + line.trim());
        	}
        	System.err.println(error[0]);
        	throw e;
		}
        final String[] ptxArray = new String[1];
        JNvrtc.nvrtcGetPTX(program, ptxArray);
        JNvrtc.nvrtcDestroyProgram(program);
        final String ptx = ptxArray[0];
        safeWrite(cache, ptx);
		return ptx;
	}

	private static String readMacros() {
		try {
			return new String(CudaProgram.class.getResourceAsStream("macros.cu").readAllBytes());
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	@Override
	public GPUKernel buildKernel(String name) {
		final CudaKernel kernel = new CudaKernel(context, stream, ptx, name);
        kernels.add(kernel);
        return kernel;
    }
	
	void release() {
		kernels.forEach(k -> k.release());
	}
	
   private final static char[] hexArray = "0123456789ABCDEF".toCharArray();
   private static String bytesToHex(byte[] bytes) {
        char[] hexChars = new char[bytes.length * 2];
        for ( int j = 0; j < bytes.length; j++ ) {
            int v = bytes[j] & 0xFF;
            hexChars[j * 2] = hexArray[v >>> 4];
            hexChars[j * 2 + 1] = hexArray[v & 0x0F];
        }
        return new String(hexChars);
    }
   
	private static String safeRead(File f) {
		try {
			return new String(Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8);
		} catch (IOException e) {
			throw new IllegalStateException(e);
		}
	}
	
	private static void safeWrite(File f, String ptx) {
		try {
			Files.write(f.toPath(), ptx.getBytes(StandardCharsets.UTF_8), StandardOpenOption.CREATE);
		} catch (IOException e) {
			throw new IllegalStateException(e);
		}
	}
	
	private static File getCacheFile(String sourceHash) {
		final String tmpDir = System.getProperty("java.io.tmpdir") + "/hs-cache";
		new File(tmpDir).mkdirs();
		return new File(tmpDir + "/" + sourceHash);
	}
}
