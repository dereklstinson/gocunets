package arrays

import (
	"unsafe"

	"github.com/dereklstinson/cuda"
)

//CUArray1d is a array that is stored in the gpu.
//The values are hidden because of the nature of gpu computing.
//As of now only use values of float32
//Also there is no garbage collection on this.  The reason being is that there
//is a small
type CUArray1d struct {
	hostdata   []float32
	devicedata cuda.DMalloc
}

//CreateCUArray1d will take the HArray3d values and load them into the device
//It will return a CUArray1d that CUArray1d is independent of the HArray3d it came from. It will
//copy the data from the passed HArray3d, but from now on they are independent of each other.
//if an error happens it will return an empty dray with the error.
func CreateCUArray1d(ray []float32) (CUArray1d, error) {
	var dray CUArray1d
	var err error
	dray.devicedata, err = cuda.LoadToDevice(ray)
	if err != nil {
		return dray, err
	}
	dray.hostdata = make([]float32, len(ray))
	for i := 0; i < len(ray); i++ {
		dray.hostdata[i] = ray[i]
	}

	return dray, nil
}

//DPTR returns the device Pointer
func (ray *CUArray1d) DPTR() unsafe.Pointer {
	return ray.devicedata.Ptr
}

//returns XYZ sizes
func (ray *CUArray1d) X() int {
	return len(ray.hostdata)
}

//Clone makes a clone of the CUArray1d.
func (ray *CUArray1d) Clone() (CUArray1d, error) {
	var dray CUArray1d
	var err error
	dray.devicedata, err = cuda.LoadToDevice(ray.hostdata)
	if err != nil {
		return dray, err
	}
	dray.hostdata = make([]float32, len(ray.hostdata))
	for i := 0; i < len(ray.hostdata); i++ {
		dray.hostdata[i] = ray.hostdata[i]
	}
	return dray, nil

}

//CloneEmpty makes a zeroed out clone of the CUArray1d.
func (ray *CUArray1d) CloneEmpty() (CUArray1d, error) {
	var dray CUArray1d
	var err error
	dray.devicedata, err = cuda.LoadToDevice(make([]float32, len(ray.hostdata)))
	if err != nil {
		return dray, err
	}
	dray.hostdata = make([]float32, len(ray.hostdata))
	return dray, nil
}
