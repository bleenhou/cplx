package eth.bram.cplx;


import jcuda.Pointer;

class CudaPointer implements GPUPointer {

	private final Pointer pointer;
	
	CudaPointer(Pointer pointer) {
		this.pointer = pointer;
	}
	
	Pointer pointer() {
		return pointer;
	}

}
