package tensor

import (
	"errors"
	"strconv"

	"github.com/dereklstinson/GoCuNets/utils"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//SetValues sets all the values in the tensor to whatever is passed. It does this by looking at the format that is held in the tensor descriptor and auto retypes it.
func (t *Volume) SetValues(handle *gocudnn.Handle, input float64) error {
	dtype, _, _, err := t.tD.GetDescrptor()

	if err != nil {
		return err
	}
	switch dtype {
	case t.thelp.Flgs.Data.Double():
		err = t.thelp.Funcs.SetTensor(handle, t.tD, t.mem, gocudnn.CDouble(input))
	case t.thelp.Flgs.Data.Float():
		err = t.thelp.Funcs.SetTensor(handle, t.tD, t.mem, gocudnn.CFloat(input))
	case t.thelp.Flgs.Data.Int32():
		err = t.thelp.Funcs.SetTensor(handle, t.tD, t.mem, gocudnn.CInt(input))
	case t.thelp.Flgs.Data.Int8():
		err = t.thelp.Funcs.SetTensor(handle, t.tD, t.mem, gocudnn.CInt8(input))
	case t.thelp.Flgs.Data.UInt8():
		err = t.thelp.Funcs.SetTensor(handle, t.tD, t.mem, gocudnn.CUInt8(input))
	default:
		return errors.New("Not supported Format to make Set All Values")
	}
	if err != nil {
		return err
	}
	return nil
}

//ScaleValues values will scale the values to the scalar passed
func (t *Volume) ScaleValues(h *gocudnn.Handle, alpha float64) error {
	dtype, _, _, err := t.tD.GetDescrptor()
	if err != nil {
		return err
	}
	switch dtype {
	case t.thelp.Flgs.Data.Double():
		return t.thelp.Funcs.ScaleTensor(h, t.tD, t.mem, gocudnn.CDouble(alpha))
	case t.thelp.Flgs.Data.Float():
		return t.thelp.Funcs.ScaleTensor(h, t.tD, t.mem, gocudnn.CFloat(alpha))
	case t.thelp.Flgs.Data.Int32():
		return t.thelp.Funcs.ScaleTensor(h, t.tD, t.mem, gocudnn.CInt(alpha))
	case t.thelp.Flgs.Data.Int8():
		err = t.thelp.Funcs.ScaleTensor(h, t.tD, t.mem, gocudnn.CInt8(alpha))
	case t.thelp.Flgs.Data.UInt8():
		err = t.thelp.Funcs.ScaleTensor(h, t.tD, t.mem, gocudnn.CUInt8(alpha))

	}
	return errors.New("Not supported Format to make zero")
}

//AddTo formula is  (t *Tensor)= alpha*(A)+beta*(t *Tensor)
//Dim max is 5. Number of dims need to be the same.  Dim size need to match or be equal to 1.
//In the later case the same value from the A tensor for the dims will be used to blend into (t *Tensor).
func (t *Volume) AddTo(h *gocudnn.Handle, A *Volume, alpha, beta float64) error {
	dtype, _, _, err := t.tD.GetDescrptor()
	if err != nil {
		return err
	}
	dtypeA, _, _, err := A.tD.GetDescrptor()
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

	return t.thelp.Funcs.AddTensor(h, a, A.tD, A.mem, b, t.tD, t.mem)
}

//LoadMem will Load the mem with
func (t *Volume) LoadMem(input gocudnn.Memer) error {

	if t.mem.ByteSize() != input.ByteSize() {
		destsize := strconv.Itoa(int(t.mem.ByteSize()))
		srcsize := strconv.Itoa(int(input.ByteSize()))
		return errors.New("LoadMem: MemSize Not same in bytes " + destsize + " " + srcsize)
	}
	kind, err := gocudnn.MemCpyDeterminer(input, t.mem)
	if err != nil {
		return prependerror("LoadMem", err)
	}

	if t.managed == true {
		return gocudnn.CudaMemCopy(t.mem, input, input.ByteSize(), gocudnn.MemcpyKindFlag{}.Default())

	}
	return gocudnn.CudaMemCopy(t.mem, input, input.ByteSize(), kind)

}
func prependerror(info string, input error) error {
	return errors.New(info + ": " + input.Error())
}

//SetRandom sets Random Value to weights
func (t *Volume) SetRandom(mean, max, fanin float64) error {
	_, dtype, dims, err := t.Properties()
	if err != nil {

		return prependerror("SetRandom", err)
	}
	vol := volume(dims)
	vol1 := int(vol)
	size, err := t.Size()
	if err != nil {
		return prependerror("SetRandom", err)
	}

	switch dtype {

	case t.thelp.Flgs.Data.Double():
		randomizedvol := make([]float64, vol)
		for i := 0; i < vol1; i++ {
			randomizedvol[i] = utils.RandWeightSet(mean, max, fanin)
		}
		ptr, err := gocudnn.MakeGoPointer(randomizedvol)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		memflag, err := gocudnn.MemCpyDeterminer(ptr, t.mem)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		return gocudnn.CudaMemCopy(t.mem, ptr, size, memflag)
	case t.thelp.Flgs.Data.Float():
		randomizedvol := make([]float32, vol)
		for i := 0; i < vol1; i++ {
			randomizedvol[i] = float32(utils.RandWeightSet(mean, max, fanin))
		}
		ptr, err := gocudnn.MakeGoPointer(randomizedvol)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		memflag, err := gocudnn.MemCpyDeterminer(ptr, t.mem)
		if err != nil {
			return prependerror("SetRandom", err)
		}

		return gocudnn.CudaMemCopy(t.mem, ptr, size, memflag)
	case t.thelp.Flgs.Data.Int32():
		if max > -1 && max < 1 {
			return errors.New("SetRandom: Max needs to be changed because it will only be zero")
		}
		randomizedvol := make([]int32, vol)
		for i := 0; i < vol1; i++ {
			randomizedvol[i] = int32(utils.RandWeightSet(mean, max, fanin))
		}
		ptr, err := gocudnn.MakeGoPointer(randomizedvol)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		memflag, err := gocudnn.MemCpyDeterminer(ptr, t.mem)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		return gocudnn.CudaMemCopy(t.mem, ptr, size, memflag)
	case t.thelp.Flgs.Data.Int8():
		if (max > -1 && max < 1) || max > 127 {
			return errors.New("Unsupported Max Value for datatype")
		}
		randomizedvol := make([]int8, vol)
		for i := 0; i < vol1; i++ {
			randomizedvol[i] = int8(utils.RandWeightSet(mean, max, fanin))
		}
		ptr, err := gocudnn.MakeGoPointer(randomizedvol)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		memflag, err := gocudnn.MemCpyDeterminer(ptr, t.mem)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		return gocudnn.CudaMemCopy(t.mem, ptr, size, memflag)
	case t.thelp.Flgs.Data.UInt8():
		if max < 1 || max > 255 {
			return errors.New("Unsupported Max Value for datatype")
		}
		randomizedvol := make([]uint8, vol)
		for i := 0; i < vol1; i++ {
			randomizedvol[i] = uint8(utils.RandWeightSet(mean, max, fanin))
		}
		ptr, err := gocudnn.MakeGoPointer(randomizedvol)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		memflag, err := gocudnn.MemCpyDeterminer(ptr, t.mem)
		if err != nil {
			return prependerror("SetRandom", err)
		}
		return gocudnn.CudaMemCopy(t.mem, ptr, size, memflag)
	}
	return errors.New("SetRandom: Unreachable Area has been reached")
}

func volume(dims []int32) int32 {
	mult := int32(1)
	for i := 0; i < len(dims); i++ {
		mult *= dims[i]
	}
	return mult
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
func (t *Tensor) Transform(h *gocudnn.Handle, A *Tensor, alpha, beta float64) error {
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
	return t.thelp.Funcs.TransformTensor(h, a, A.tD, A.mem, b, t.tD, t.mem)
}
*/
//func (t *Tensor) AddAll()
