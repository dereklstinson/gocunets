//Package tensor is used to make tensors by using gocudnn.  It is currently not supporting what I call the "EX" functions.
//because the Tensor struct is also going to be carrying a filter descripter.  Also I call it "EX" functions loosly, because I think
//there is a miss labeling of the function names in cudnn. Basicly it is the set tensor fuctions that don't include the format and include
//the strides
package tensor

import (
	"errors"
	"fmt"

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

	newmem, err := gocudnn.Malloc(t.mem.ByteSize())
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

//Destroy will release the memory of the tensor
func (t *Volume) Destroy() error {
	return destroy(t)
}
