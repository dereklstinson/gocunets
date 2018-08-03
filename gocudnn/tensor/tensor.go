//Package tensor is used to make tensors by using gocudnn.  It is currently not supporting what I call the "EX" functions.
//because the Tensor struct is also going to be carrying a filter descripter.  Also I call it "EX" functions loosly, because I think
//there is a miss labeling of the function names in cudnn. Basicly it is the set tensor fuctions that don't include the format and include
//the strides
package tensor

import "C"
import (
	"errors"
	"fmt"
	"image"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Volume holds both a gocudnn.TensorD and gocudnn.FilterD and the allocated memory associated with it
type Volume struct {
	tD      *gocudnn.TensorD
	fD      *gocudnn.FilterD
	dtype   gocudnn.DataType
	propnan gocudnn.PropagationNAN
	mem     gocudnn.Memer
	fmt     gocudnn.TensorFormat
	thelp   gocudnn.Tensor
	fhelp   gocudnn.Filter
	ophelp  gocudnn.OpTensor
	managed bool
	//scalar gocudnn.CScalar
}

//SetPropNan will change the default nan propigation flag from PropNanNon to PropNaN
func (t *Volume) SetPropNan() {
	t.propnan = t.thelp.Flgs.NaN.PropagateNan()
}

//SetNotPropNan will set the nan propigation flag to NotPropigationNan (NotPropigationNan is default)
func (t *Volume) SetNotPropNan() {
	t.propnan = t.thelp.Flgs.NaN.NotPropagateNan()
}

//Flags returns a struct that passes gocudnn flags through methods used in building the tensor
func Flags() gocudnn.TensorFlags {
	return gocudnn.TensorFlags{}
}

//Build creates a tensor and mallocs the memory for the tensor
func Build(fmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32, managed bool) (*Volume, error) {
	var thelper gocudnn.Tensor
	var fhelper gocudnn.Filter
	if len(dims) < 4 {
		return nil, errors.New("Dims less than 4. Create A 4 dim Tensor and set dims not needed to 1")
	}
	var newmemer *gocudnn.Malloced
	var tens *gocudnn.TensorD
	var filts *gocudnn.FilterD
	var err error
	if len(dims) > 4 {
		tens, err = thelper.NewTensorNdDescriptorEx(fmt, dtype, dims)
		if err != nil {
			return nil, err
		}
		filts, err = fhelper.NewFilterNdDescriptor(dtype, fmt, dims)
		if err != nil {
			tens.DestroyDescriptor()
			return nil, err
		}
		size, err := tens.GetSizeInBytes()
		if err != nil {
			tens.DestroyDescriptor()
			filts.DestroyDescriptor()
			return nil, err
		}
		if managed == true {
			newmemer, err = gocudnn.MallocManaged(size, gocudnn.ManagedMemFlag{}.Global())
			if err != nil {

				tens.DestroyDescriptor()
				filts.DestroyDescriptor()
				return nil, err
			}

		} else {

			newmemer, err = gocudnn.Malloc(size)
			if err != nil {

				tens.DestroyDescriptor()
				filts.DestroyDescriptor()
				return nil, err

			}

		}

	} else {

		tens, err = thelper.NewTensor4dDescriptor(dtype, fmt, dims)
		if err != nil {
			return nil, err
		}
		filts, err = fhelper.NewFilter4dDescriptor(dtype, fmt, dims)
		if err != nil {
			tens.DestroyDescriptor()
			return nil, err
		}
		size, err := tens.GetSizeInBytes()
		if err != nil {
			tens.DestroyDescriptor()
			filts.DestroyDescriptor()
			return nil, err
		}
		if managed == true {

			newmemer, err = gocudnn.MallocManaged(size, gocudnn.ManagedMemFlag{}.Global())
			if err != nil {

				tens.DestroyDescriptor()
				filts.DestroyDescriptor()
				return nil, err

			}

		} else {
			newmemer, err = gocudnn.Malloc(size)
			if err != nil {

				tens.DestroyDescriptor()
				filts.DestroyDescriptor()
				return nil, err

			}

		}

	}

	return &Volume{
		tD:    tens,
		fD:    filts,
		mem:   newmemer,
		fmt:   fmt,
		dtype: dtype,
	}, nil

}

//TD returns the tensor descriptor for Tensor
func (t *Volume) TD() *gocudnn.TensorD {
	return t.tD
}

//FD returns the filter descriptor for Tensor
func (t *Volume) FD() *gocudnn.FilterD {
	return t.fD
}

//Memer returns the Memer for Tensor
func (t *Volume) Memer() gocudnn.Memer {
	return t.mem
}

//Size returns the size in bytes in type gocudnn.SizeT
func (t *Volume) Size() (gocudnn.SizeT, error) {
	return t.tD.GetSizeInBytes()
}

//Properties returns the properties of the tensor
func (t *Volume) Properties() (gocudnn.TensorFormat, gocudnn.DataType, []int32, error) {
	a, b, _, err := t.tD.GetDescrptor()
	if err != nil {
		return t.fmt, a, b, err
	}
	return t.fmt, a, b, nil
}

//ZeroClone returns a zero clone of the the memory
func (t *Volume) ZeroClone(handle *gocudnn.Handle) (*Volume, error) {

	if t.tD == nil || t.fD == nil || t.mem == nil {
		return nil, errors.New("Tensor is nil")
	}
	dtype, dims, strides, err := t.tD.GetDescrptor()
	if err != nil {
		return nil, err
	}

	var filt *gocudnn.FilterD
	var tens *gocudnn.TensorD
	if len(strides) > 0 {
		if len(dims) > 4 {
			tens, err = t.thelp.NewTensorNdDescriptor(dtype, dims, strides)
		} else {
			tens, err = t.thelp.NewTensor4dDescriptorEx(dtype, dims, strides)
		}
	} else {
		if len(dims) > 4 {
			tens, err = t.thelp.NewTensorNdDescriptorEx(t.fmt, dtype, dims)
		} else {
			tens, err = t.thelp.NewTensor4dDescriptor(dtype, t.fmt, dims)
		}
	}
	if err != nil {
		return nil, err
	}
	if len(dims) > 4 {
		filt, err = t.fhelp.NewFilterNdDescriptor(dtype, t.fmt, dims)
	} else {
		filt, err = t.fhelp.NewFilter4dDescriptor(dtype, t.fmt, dims)
	}
	if err != nil {
		return nil, err
	}
	var newmem *gocudnn.Malloced
	if t.managed == true {
		newmem, err = gocudnn.MallocManaged(t.mem.ByteSize(), gocudnn.ManagedMemFlag{}.Global())
	} else {
		newmem, err = gocudnn.Malloc(t.mem.ByteSize())
	}

	if err != nil {
		return nil, err
	}

	switch dtype {
	case t.thelp.Flgs.Data.Double():
		err = t.thelp.Funcs.SetTensor(handle, tens, newmem, gocudnn.CDouble(0))
	case t.thelp.Flgs.Data.Float():
		err = t.thelp.Funcs.SetTensor(handle, tens, newmem, gocudnn.CFloat(0))
	case t.thelp.Flgs.Data.Int32():
		err = t.thelp.Funcs.SetTensor(handle, tens, newmem, gocudnn.CInt(0))
	default:
		return nil, errors.New("Not supported Format to make zero")
	}
	if err != nil {
		return nil, err
	}
	return &Volume{tD: tens, fD: filt, mem: newmem, fmt: t.fmt}, nil
}

func destroy(t *Volume) error {
	var flag bool

	err1 := t.tD.DestroyDescriptor()
	if err1 != nil {
		flag = true
	}
	err2 := t.fD.DestroyDescriptor()
	if err2 != nil {
		flag = true
	}
	err3 := t.mem.Free()
	if err3 != nil {
		flag = true
	}
	if flag == true {
		return fmt.Errorf("error::TensorD: %sFilterD: %sMemory: %s", err1, err2, err3)
	}
	return nil
}

//arraysize will return the size of the array and will return 0 if unsupported type is used.
func arraysize(dtype gocudnn.DataType, size gocudnn.SizeT) int {
	var flg gocudnn.DataTypeFlag
	x := int(size)
	switch dtype {
	case flg.Double():
		return x / 8
	case flg.Float():
		return x / 4
	case flg.Int32():
		return x / 4
	case flg.UInt8():
		return x
	case flg.Int8():
		return x
	default:
		return 0
	}
}

func (t *Volume) PrintUnifiedMem() error {
	kind := gocudnn.MemcpyKindFlag{}.Default()
	return t.printmem(kind)
}

func (t *Volume) printmem(kind gocudnn.MemcpyKind) error {
	var flg gocudnn.DataTypeFlag
	sib := t.mem.ByteSize()
	as := arraysize(t.dtype, sib)

	switch t.dtype {
	case flg.Double():
		array := make([]C.double, as)
		ptr, err := gocudnn.MakeGoPointer(array)
		if err != nil {
			return err
		}
		err = gocudnn.CudaMemCopy(ptr, t.mem, sib, kind)
		if err != nil {
			return err
		}
		fmt.Println(array)
	case flg.Float():
		array := make([]C.float, as)
		ptr, err := gocudnn.MakeGoPointer(array)

		if err != nil {
			return err
		}
		err = gocudnn.CudaMemCopy(ptr, t.mem, sib, kind)
		if err != nil {
			return err
		}
		fmt.Println(array)
	case flg.Int32():
		array := make([]C.int, as)
		ptr, err := gocudnn.MakeGoPointer(array)

		if err != nil {
			return err
		}
		err = gocudnn.CudaMemCopy(ptr, t.mem, sib, kind)
		if err != nil {
			return err
		}
		fmt.Println(array)
	case flg.UInt8():
		array := make([]C.uchar, as)
		ptr, err := gocudnn.MakeGoPointer(array)

		if err != nil {
			return err
		}
		err = gocudnn.CudaMemCopy(ptr, t.mem, sib, kind)
		if err != nil {
			return err
		}
		fmt.Println(array)
	case flg.Int8():
		array := make([]C.schar, as)
		ptr, err := gocudnn.MakeGoPointer(array)
		if err != nil {
			return err
		}
		err = gocudnn.CudaMemCopy(ptr, t.mem, sib, kind)
		if err != nil {
			return err
		}
		fmt.Println(array)
	default:
		return errors.New("Unsupoorted Format")
	}

	return nil
}
func arraysizefromdims(dims []int32) int {
	mult := 1
	for i := 0; i < len(dims); i++ {
		mult *= int(dims[i])
	}
	return mult
}

func unnormfloat32(input []float32) {
	imin := float32(999999999.0)
	imax := -float32(99999999.0)
	for i := 0; i < len(input); i++ {
		if imin > input[i] {
			imin = input[i]
		}
		if imax < input[i] {
			imax = input[i]
		}
	}
	boundrange := max - min
	irange := imax - imin
	ratio := float32(boundrange) / irange
	for i := 0; i < len(input); i++ {
		input[i] -= (imin - float32(min))
		input[i] *= ratio

	}

}
func unnormfloat64(min, max int, input []float64) {
	imin := float64(999999999.0)
	imax := -float64(99999999.0)
	for i := 0; i < len(input); i++ {
		if imin > input[i] {
			imin = input[i]
		}
		if imax < input[i] {
			imax = input[i]
		}
	}
	boundrange := max - min
	irange := imax - imin
	ratio := float64(boundrange) / irange
	for i := 0; i < len(input); i++ {
		input[i] -= (imin - float64(min))
		input[i] *= ratio

	}
}
func unnormalizeint32(min, max int, input []int32) {
	imin := float32(999999999.0)
	imax := -float32(99999999.0)
	for i := 0; i < len(input); i++ {
		if imin > float32(input[i]) {
			imin = float32(input[i])
		}
		if imax < float32(input[i]) {
			imax = float32(input[i])
		}
	}
	boundrange := max - min
	irange := imax - imin
	ratio := float32(boundrange) / irange
	for i := 0; i < len(input); i++ {
		input[i] -= int32((imin - float32(min)))
		input[i] = int32(float32(input[i]) * ratio)

	}

}
func unnormalize(min, max int, input interface{}) {
	switch x := input.(type) {
	case []float32:
		unnormfloat32(min, max, x)
	case []int32:
		unnormalizeint32(min, max, x)
	case []float64:
		unnormfloat64(min, max, x)
	case []uint8:

	case []int8:

	}

}
func touint8(input interface{}) []uint8 {
	switch x := input.(type) {

	case []float32:
	case []int32:
	case []float64:
	case []uint8:
		return x
	case []int8:

	}
	return []uint8{}
}

func (t *Volume) convert() ([][]image.Image, error) {
	frmt, dtype, dims, err := t.Properties()
	var tf gocudnn.TensorFormatFlag
	var dt gocudnn.DataTypeFlag
	if err != nil {
		return nil, err
	}

	switch dtype {
	case dt.Double():

	case dt.Float():
		slice := make([]float32, arraysizefromdims(dims))
		err = t.mem.FillSlice(slice)
		if err != nil {
			return nil, err
		}
		switch frmt {
		case tf.NCHW():
		case tf.NHWC():
		case tf.NCHWvectC():
		}

	case dt.Int32():

	case dt.Int8():

	case dt.UInt8():

	}
	var rect image.Rectangle

	rect.Min.X = 0
	rect.Min.Y = 0
	rect.Max.X = 28
	rect.Max.Y = 28
	img := image.NewUniform(rect)

	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			pix := uint8(data.Data[i*28+j])

			img.Set(j, i, gray.RGBA{pix, pix, pix, 255})
		}
	}
	return nil, img
}

//PrintDeviceMem.  Kind of a shortcut function. I would like to build a more extensive function in the future where it would just know what to do without much user input.  It would use this function so it is not a waste.
func (t *Volume) PrintDeviceMem() error {
	kind := gocudnn.MemcpyKindFlag{}.DeviceToHost()
	return t.printmem(kind)
}

//Destroy will release the memory of the tensor
func (t *Volume) Destroy() error {
	return destroy(t)
}
