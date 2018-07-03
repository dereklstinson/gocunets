//Package layers contains shared things between layers.  It also contains functions that will be supplimental to cudnn.
package layers

import (
	"fmt"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//IO is an all purpose struct that contains an x tensor and a dx tensor used for training
type IO struct {
	x    *tensor.Tensor
	dx   *tensor.Tensor
	dims []int32
}

func (i *IO) Properties() (gocudnn.TensorFormat, gocudnn.DataType, []int32, error) {
	return i.x.Properties()
}

//DTensor returns d tensor
func (i *IO) DTensor() *tensor.Tensor {
	return i.dx
}

//Tensor returns the tensor
func (i *IO) Tensor() *tensor.Tensor {
	return i.x
}

//Mem returns the main memery //Legacy func will go away
func (i *IO) Mem() gocudnn.Memer {
	return i.x.Memer()
}

//DMem returns the error backprop memory for the tensor memory  //Legacy func will go away
func (i *IO) DMem() gocudnn.Memer {
	return i.dx.Memer()
}

//BuildIO builds an IO
func BuildIO(fmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32) (*IO, error) {
	x, err := tensor.Create(fmt, dtype, dims)
	if err != nil {
		x.Destroy()
		return nil, err
	}
	dx, err := tensor.Create(fmt, dtype, dims)
	if err != nil {
		x.Destroy()
		dx.Destroy()
		return nil, err
	}
	return &IO{
		x:  x,
		dx: dx,
	}, nil
}

//Destroy frees all the memory assaciated with the tensor inside of IO
func (i *IO) Destroy() error {
	var flag bool
	err := i.dx.Destroy()
	if err != nil {
		flag = true
	}
	err1 := i.x.Destroy()
	if err1 != nil {
		flag = true
	}
	if flag == true {
		return fmt.Errorf("error:x: %s,dx: %s", err, err1)
	}
	return nil
}

/*
//LoadMem Replaces The memory on the device.
func (i *IO) LoadMem(mem gocudnn.Memer, kind gocudnn.MemcpyKind) error {
	size, err := i.desc.GetSizeInBytes()
	if err != nil {
		return err
	}
	if size != mem.ByteSize() {
		return errors.New("Memory Size doesn't Match Descriptor")
	}
	gocudnn.CudaMemCopy(i.mem, mem, size, kind)
	return nil
}
*/
