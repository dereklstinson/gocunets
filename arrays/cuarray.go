package arrays

import (
	"github.com/dereklstinson/cuda"
)

//CUArray3d is a array that is stored in the gpu.
//The values are hidden because of the nature of gpu computing.
//As of now only use values of float32
//Also there is no garbage collection on this.  The reason being is that there
//is a small
type CUArray3d struct {
	hostdata   []float32
	devicedata cuda.DMalloc
	x, y, z    int
}

//CreateCUArray3d will take the HArray3d values and load them into the device
//It will return a CUArray3d that CUArray3d is independent of the HArray3d it came from. It will
//copy the data from the passed HArray3d, but from now on they are independent of each other.
//if an error happens it will return an empty dray with the error.
func CreateCUArray3d(ray HArray3d) (CUArray3d, error) {
	var dray CUArray3d
	var err error
	dray.devicedata, err = cuda.LoadToDevice(ray.Data)
	if err != nil {
		return dray, err
	}
	dray.x = ray.X
	dray.y = ray.Y
	dray.z = ray.Z
	dray.hostdata = ray.copydata()

	return dray, nil
}

//Clone makes a clone of the CUArray3d.
func (ray *CUArray3d) Clone() (CUArray3d, error) {
	var dray CUArray3d
	var err error
	dray.devicedata, err = cuda.LoadToDevice(ray.hostdata)
	if err != nil {
		return dray, err
	}
	dray.x = ray.x
	dray.y = ray.y
	dray.z = ray.z
	dray.hostdata = make([]float32, len(ray.hostdata))
	for i := 0; i < len(ray.hostdata); i++ {
		dray.hostdata[i] = ray.hostdata[i]
	}
	return dray, nil

}

//CloneEmpty makes a zeroed out clone of the CUArray3d.
func (ray *CUArray3d) CloneEmpty() (CUArray3d, error) {
	var dray CUArray3d
	var err error
	dray.devicedata, err = cuda.LoadToDevice(make([]float32, len(ray.hostdata)))
	if err != nil {
		return dray, err
	}
	dray.x = ray.x
	dray.y = ray.y
	dray.z = ray.z
	dray.hostdata = make([]float32, len(ray.hostdata))
	return dray, nil
}
