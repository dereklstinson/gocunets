package tensor

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//SetValues sets all the values in the tensor to whatever is passed. It does this by looking at the format that is held in the tensor descriptor and auto retypes it.
func (t *Volume) SetValues(handle *cudnn.Handler, input float64) error {
	dtype, _, _, err := t.current.tD.GetDescrptor()

	if err != nil {
		return err
	}
	switch dtype {
	case t.thelp.Flgs.Data.Double():
		err = t.thelp.SetTensor(handle.Cudnn(), t.current.tD, t.memgpu, gocudnn.CDouble(input))
	case t.thelp.Flgs.Data.Float():
		err = t.thelp.SetTensor(handle.Cudnn(), t.current.tD, t.memgpu, gocudnn.CFloat(input))
	case t.thelp.Flgs.Data.Int32():
		err = t.thelp.SetTensor(handle.Cudnn(), t.current.tD, t.memgpu, gocudnn.CInt(input))
	case t.thelp.Flgs.Data.Int8():
		err = t.thelp.SetTensor(handle.Cudnn(), t.current.tD, t.memgpu, gocudnn.CInt8(input))
	case t.thelp.Flgs.Data.UInt8():
		err = t.thelp.SetTensor(handle.Cudnn(), t.current.tD, t.memgpu, gocudnn.CUInt8(input))
	default:
		return errors.New("Not supported Format to make Set All Values")
	}
	if err != nil {
		return err
	}
	return nil
}

//ScaleValues values will scale the values to the scalar passed
func (t *Volume) ScaleValues(handle *cudnn.Handler, alpha float64) error {
	dtype, _, _, err := t.current.tD.GetDescrptor()
	if err != nil {
		return err
	}
	switch dtype {
	case t.thelp.Flgs.Data.Double():
		return t.thelp.ScaleTensor(handle.Cudnn(), t.current.tD, t.memgpu, gocudnn.CDouble(alpha))
	case t.thelp.Flgs.Data.Float():
		return t.thelp.ScaleTensor(handle.Cudnn(), t.current.tD, t.memgpu, gocudnn.CFloat(alpha))
	case t.thelp.Flgs.Data.Int32():
		return t.thelp.ScaleTensor(handle.Cudnn(), t.current.tD, t.memgpu, gocudnn.CInt(alpha))
	case t.thelp.Flgs.Data.Int8():
		err = t.thelp.ScaleTensor(handle.Cudnn(), t.current.tD, t.memgpu, gocudnn.CInt8(alpha))
	case t.thelp.Flgs.Data.UInt8():
		err = t.thelp.ScaleTensor(handle.Cudnn(), t.current.tD, t.memgpu, gocudnn.CUInt8(alpha))

	}
	return errors.New("Not supported Format to make zero")
}

//AddTo formula is  (t *Tensor)= alpha*(A)+beta*(t *Tensor)
//Dim max is 5. Number of dims need to be the same.  Dim size need to match or be equal to 1.
//In the later case the same value from the A tensor for the dims will be used to blend into (t *Tensor).
func (t *Volume) AddTo(handle *cudnn.Handler, A *Volume, Amultiplier, tmultiplier float64) error {
	alpha := Amultiplier
	beta := tmultiplier
	dtype, _, _, err := t.current.tD.GetDescrptor()
	if err != nil {
		return err
	}
	dtypeA, _, _, err := A.current.tD.GetDescrptor()
	if err != nil {
		return err
	}
	if dtype != dtypeA {
		return errors.New("Datatypes Don't Match for Scalar")
	}
	var a gocudnn.CScalar
	var b gocudnn.CScalar
	switch dtype {
	case t.thelp.Flgs.Data.Double():
		a = gocudnn.CDouble(alpha)
		b = gocudnn.CDouble(beta)
	case t.thelp.Flgs.Data.Float():
		a = gocudnn.CFloat(alpha)
		b = gocudnn.CFloat(beta)
	case t.thelp.Flgs.Data.Int32():
		a = gocudnn.CInt(alpha)
		b = gocudnn.CInt(beta)
	case t.thelp.Flgs.Data.Int8():
		a = gocudnn.CInt8(alpha)
		b = gocudnn.CInt8(beta)
	case t.thelp.Flgs.Data.UInt8():
		a = gocudnn.CUInt8(alpha)
		b = gocudnn.CUInt8(beta)

	default:
		return errors.New("AddTo: Not supported Format to make zero")
	}

	return t.thelp.AddTensor(handle.Cudnn(), a, A.current.tD, A.memgpu, b, t.current.tD, t.memgpu)
}

//LoadMem will Load the mem with
func (t *Volume) LoadMem(handle *cudnn.Handler, input gocudnn.Memer) error {

	if t.CurrentSizeT() != cudnn.SizeT(input.ByteSize()) {
		destsize := strconv.Itoa(int(t.memgpu.ByteSize()))
		srcsize := strconv.Itoa(int(input.ByteSize()))
		return errors.New("LoadMem: MemSize Not same in bytes " + destsize + " " + srcsize)
	}
	kind, err := gocudnn.MemCpyDeterminer(input, t.memgpu)
	if err != nil {
		return prependerror("LoadMem", err)
	}

	if handle.Unified() {
		return gocudnn.CudaMemCopy(t.memgpu, input, input.ByteSize(), gocudnn.MemcpyKindFlag{}.Default())

	}
	return gocudnn.CudaMemCopy(t.memgpu, input, input.ByteSize(), kind)

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

	switch dtype.Cu() {

	case t.thelp.Flgs.Data.Double():
		randomizedvol := make([]float64, vol)
		for i := 0; i < vol1; i++ {
			randomizedvol[i] = utils.RandomFloat64(float64(min), float64(max))
		}
		ptr, err := gocudnn.MakeGoPointer(randomizedvol)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		return t.LoadMem(handle, ptr)
	case t.thelp.Flgs.Data.Float():
		randomizedvol := make([]float32, vol)
		for i := 0; i < vol1; i++ {
			randomizedvol[i] = utils.RandomFloat32(min, max)
		}
		ptr, err := gocudnn.MakeGoPointer(randomizedvol)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		return t.LoadMem(handle, ptr)

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

	switch dtype.Cu() {

	case t.thelp.Flgs.Data.Double():
		randomizedvol := make([]float64, vol)
		for i := 0; i < vol1; i++ {
			randomizedvol[i] = utils.RandWeightSet(mean, max, fanin)
		}
		ptr, err := gocudnn.MakeGoPointer(randomizedvol)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		memflag, err := gocudnn.MemCpyDeterminer(ptr, t.memgpu)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		fmt.Println("memcopyflag-double")
		return gocudnn.CudaMemCopy(t.memgpu, ptr, size.Cu(), memflag)
	case t.thelp.Flgs.Data.Float():
		fmt.Println("Going Into Float")
		randomizedvol := make([]float32, vol)
		for i := 0; i < vol1; i++ {

			randomizedvol[i] = float32(utils.RandWeightSet(mean, max, fanin))
		}
		fmt.Println("Making Go Pointer")
		ptr, err := gocudnn.MakeGoPointer(randomizedvol)
		if err != nil {
			fmt.Println("GOPOINTER ERROR")
			return prependerror("SetRandom", err)
		}
		//memflag, err := gocudnn.MemCpyDeterminer(ptr, t.memgpu)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		err = gocudnn.CudaMemCopy(t.memgpu, ptr, size.Cu(), gocudnn.MemcpyKindFlag{}.Default())
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
	case t.thelp.Flgs.Data.Double():
		a = gocudnn.CDouble(alpha)
		b = gocudnn.CDouble(beta)
	case t.thelp.Flgs.Data.Float():
		a = gocudnn.CFloat(alpha)
		b = gocudnn.CFloat(beta)
	case t.thelp.Flgs.Data.Int32():
		a = gocudnn.CInt(alpha)
		b = gocudnn.CInt(beta)
	default:
		return errors.New("Not supported Format to make zero")
	}
	return t.thelp.TransformTensor(h, a, A.tD, A.mem, b, t.tD, t.mem)
}
*/
//func (t *Tensor) AddAll()
