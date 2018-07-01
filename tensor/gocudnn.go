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

//Tensor holds both a gocudnn.TensorD and gocudnn.FilterD and the allocated memory associated with it
type Tensor struct {
	tD    *gocudnn.TensorD
	fD    *gocudnn.FilterD
	mem   gocudnn.Memer
	fmt   gocudnn.TensorFormat
	thelp gocudnn.Tensor
	fhelp gocudnn.Filter
}

//Flags returns a struct that passes gocudnn flags through methods used in building the tensor
func Flags() gocudnn.TensorFlags {
	return gocudnn.TensorFlags{}
}

//Create creates a tensor and mallocs the memory for the tensor
func Create(fmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32) (*Tensor, error) {
	var thelper gocudnn.Tensor
	var fhelper gocudnn.Filter
	if len(dims) < 4 {
		return nil, errors.New("Dims less than 4. Create A 4 dim Tensor and set dims not needed to 1")
	}

	if len(dims) > 4 {

	}

	tens, err := thelper.NewTensor4dDescriptor(dtype, fmt, dims)
	if err != nil {
		return nil, err
	}
	filts, err := fhelper.NewFilter4dDescriptor(dtype, fmt, dims)
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

	newmemer, err := gocudnn.MallocManaged(size, gocudnn.ManagedMemFlag{}.Global())
	if err != nil {
		newmemer, err = gocudnn.Malloc(size)
		if err != nil {
			tens.DestroyDescriptor()
			filts.DestroyDescriptor()
			return nil, err
		}

	}
	return &Tensor{
		tD:  tens,
		fD:  filts,
		mem: newmemer,
		fmt: fmt,
	}, nil

}

//ZeroClone returns a zero clone of the the memory
func (t *Tensor) ZeroClone(handle *gocudnn.Handle) (*Tensor, error) {

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
	return &Tensor{tD: tens, fD: filt, mem: newmem, fmt: t.fmt}, nil
}

//SetAllValues sets all the values in the tensor to whatever is passed. It does this by looking at the format that is held in the tensor descriptor and auto retypes it.
func (t *Tensor) SetAllValues(handle *gocudnn.Handle, input float64) error {
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
	default:
		return errors.New("Not supported Format to make Set All Values")
	}
	if err != nil {
		return err
	}
	return nil
}
func destroy(t *Tensor) error {
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
		return fmt.Errorf("Error:: TensorD: %s,FilterD: %s,Memory: %s", err1, err2, err3)
	}
	return nil
}

//Destroy will release the memory of the tensor
func (t *Tensor) Destroy() error {
	return destroy(t)
}

//func (t *Tensor) AddAll()
