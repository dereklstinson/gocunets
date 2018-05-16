package mat

import (
	"github.com/dereklstinson/cuda"
)

//CUmat3d is a matrix that is stored in the gpu.
//The values are hidden because of the nature of gpu computing.
//As of now only use values of float32
//Also there is no garbage collection on this.  The reason being is that there
//is a small
type CUmat3d struct {
	hostdata   []float32
	devicedata cuda.DMalloc
	x, y, z    int
}

//CreateCUmat3d will take the Hmat3d values and load them into the device
//It will return a CUmat3d that CUmat3d is independent of the Hmat3d it came from. It will
//copy the data from the passed Hmat3d, but from now on they are independent of each other.
//if an error happens it will return an empty dmat with the error.
func CreateCUmat3d(mat Hmat3d) (CUmat3d, error) {
	var dmat CUmat3d
	var err error
	dmat.devicedata, err = cuda.LoadToDevice(mat.Data)
	if err != nil {
		return dmat, err
	}
	dmat.x = mat.X
	dmat.y = mat.Y
	dmat.z = mat.Z
	dmat.hostdata = mat.copydata()

	return dmat, nil
}

//Clone makes a clone of the CUmat3d.
func (mat *CUmat3d) Clone() (CUmat3d, error) {
	var dmat CUmat3d
	var err error
	dmat.devicedata, err = cuda.LoadToDevice(mat.hostdata)
	if err != nil {
		return dmat, err
	}
	dmat.x = mat.x
	dmat.y = mat.y
	dmat.z = mat.z
	dmat.hostdata = make([]float32, len(mat.hostdata))
	for i := 0; i < len(mat.hostdata); i++ {
		dmat.hostdata[i] = mat.hostdata[i]
	}
	return dmat, nil

}
