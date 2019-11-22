package tensor

import (
	"errors"
	"fmt"
	"github.com/dereklstinson/half"
	"strconv"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/cutil"
)

//SetValues sets all the values in the tensor to whatever is passed. It does this by looking at the format that is held in the tensor descriptor and auto retypes it.
//Documentation states that you have to pass a value that is the same type as the DataType.  input is typecase
func (t *Volume) SetValues(handle *cudnn.Handler, input float64) error {

	return gocudnn.SetTensor(handle.Cudnn(), t.current.tD, t.memgpu, input)
}

//ScaleValues values will scale the values to the scalar passed
func (t *Volume) ScaleValues(handle *cudnn.Handler, alpha float64) error {

	return gocudnn.ScaleTensor(handle.Cudnn(), t.current.tD, t.memgpu, alpha)

}

//AddTo formula is  (t *Tensor)= alpha*(A)+beta*(t *Tensor)
//Dim max is 5. Number of dims need to be the same.  Dim size need to match or be equal to 1.
//In the later case the same value from the A tensor for the dims will be used to blend into (t *Tensor).
func (t *Volume) AddTo(handle *cudnn.Handler, A *Volume, Ascalar, tscalar float64) error {

	return gocudnn.AddTensor(handle.Cudnn(), Ascalar, A.current.tD, A.memgpu, tscalar, t.current.tD, t.memgpu)
}

//LoadMem will Load the volume with the inputed mem.  Input mem with the size of size
func (t *Volume) LoadMem(handle *cudnn.Handler, input cutil.Mem, size uint) error {

	if t.CurrentSizeT() != size {
		fmt.Println("Dims of mem is: ", t.Dims())

		println("currentsize vs input size", t.CurrentSizeT(), size)
		destsize := strconv.Itoa(int(t.memgpu.TotalBytes()))
		currentsize := strconv.Itoa(int(t.CurrentSizeT()))

		return errors.New("LoadMem: MemSize Not same in bytes " + destsize + "  " + currentsize)
	}
	err := nvidia.Memcpy(t.memgpu, input, t.CurrentSizeT())
	if err != nil {
		fmt.Println("Loading Mem checking ointers", t.memgpu, input)
		return err
	}
	return nil
}
func prependerror(info string, input error) error {
	return errors.New(info + ": " + input.Error())
}

//SetRandomNormal sets random numbers for values in volume
func (t *Volume) SetRandomNormal(handle *cudnn.Handler, min, max float32) error {

	_, dtype, dims, err := t.Properties()
	if err != nil {

		return prependerror("SetRandomNormal", err)
	}
	vol := utils.FindVolumeInt32(dims, nil)
	vol1 := int(vol)

	if err != nil {
		return prependerror("SetRandomNormal", err)
	}
	var fflg gocudnn.DataType
	switch dtype {

	case fflg.Double():
		randomizedvol := make([]float64, vol)
		for i := 0; i < vol1; i++ {
			randomizedvol[i] = utils.RandomFloat64(float64(min), float64(max))
		}
		ptr, err := gocu.MakeGoMem(randomizedvol)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		return t.LoadMem(handle, ptr, uint(vol*8))
	case fflg.Float():
		randomizedvol := make([]float32, vol)
		for i := 0; i < vol1; i++ {
			randomizedvol[i] = utils.RandomFloat32(min, max)
		}
		ptr, err := gocu.MakeGoMem(randomizedvol)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		err = t.LoadMem(handle, ptr, uint(vol*4))
		if err != nil {
			fmt.Println("error in loading mem in set rand normoal")
		}
		return err

	}
	return errors.New("SetRandom: Unreachable Area has been reached")
}

//SetRandom sets Random Value to weights Double and Float datatype only supported
func (t *Volume) SetRandom(handle *cudnn.Handler, mean, max, fanin float64) error {
	_, dtype, dims, err := t.Properties()
	if err != nil {

		return prependerror("SetRandom", err)
	}
	vol := utils.FindVolumeInt32(dims, nil)
	vol1 := int(vol)
	size := t.CurrentSizeT()
	var fflg gocudnn.DataType
	switch dtype {

	case fflg.Double():
		randomizedvol := make([]float64, vol)
		for i := 0; i < vol1; i++ {
			randomizedvol[i] = utils.RandWeightSet(mean, max, fanin)
		}
		ptr, err := gocu.MakeGoMem(randomizedvol)
		if err != nil {
			return prependerror("SetRandom", err)
		}

		return nvidia.Memcpy(t.memgpu, ptr, size)
	case fflg.Float():

		randomizedvol := make([]float32, vol)
		for i := 0; i < vol1; i++ {

			randomizedvol[i] = float32(utils.RandWeightSet(mean, max, fanin))
		}

		ptr, err := gocu.MakeGoMem(randomizedvol)
		if err != nil {
			return prependerror("SetRandom", err)
		}

		err = nvidia.Memcpy(t.memgpu, ptr, size)
		if err != nil {
			fmt.Println("Size Value is ", size)
			fmt.Println("Size of vol is ", vol)
			fmt.Println("Vol * 4 is ", vol*4)
			fmt.Println("t.memgpu is", t.memgpu)
			return err
		}
		return nil
	case fflg.Half():
		randomizedvol := make([]half.Float16, vol)
		for i := 0; i < vol1; i++ {
			x := float32(utils.RandWeightSet(mean, max, fanin))
			randomizedvol[i] = half.NewFloat16(x)
		}

		ptr, err := gocu.MakeGoMem(randomizedvol)
		if err != nil {
			return prependerror("SetRandom", err)
		}

		err = nvidia.Memcpy(t.memgpu, ptr, size)
		if err != nil {
			fmt.Println("Size Value is ", size)
			fmt.Println("Size of vol is ", vol)
			fmt.Println("Vol * 4 is ", vol*4)
			fmt.Println("t.memgpu is", t.memgpu)
			return err
		}
		return nil
	}
	return errors.New("SetRandom: Unreachable Area has been reached")
}

//Transform tensor
/*
From the SDK Documentation:
This function copies the scaled data from one tensor to another tensor with a different layout.
Those descriptors need to have the same dimensions but not necessarily the same strides.
The input and output tensors must not overlap in any way (i.e., tensors cannot be transformed in place).
This function can be used to convert a tensor with an unsupported format to a supported one.

my guess in what this does is change the format like NCHW to NHWC of t,
This is probably an EX function
*/
/*
func (t *Tensor) Transform(handle *cudnn.Handler, A *Tensor, alpha, beta float64) error {
	dtypeA, dims, _, err := A.tD.GetDescrptor()
	if err != nil {
		return err
	}
	var a gocudnn.CScalar
	var b gocudnn.CScalar
	switch dtypeA {
	case fflg.Double():
		a = gocudnn.CDouble(alpha)
		b = gocudnn.CDouble(beta)
	case fflg.Float():
		a = gocudnn.CFloat(alpha)
		b = gocudnn.CFloat(beta)
	case fflg.Int32():
		a = gocudnn.CInt(alpha)
		b = gocudnn.CInt(beta)
	default:
		return errors.New("Not supported Format to make zero")
	}
	return t.thelp.TransformTensor(h, a, A.tD, A.mem, b, t.tD, t.mem)
}
*/
//func (t *Tensor) AddAll()
