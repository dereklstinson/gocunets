//Package layers contains shared things between layers.  It also contains functions that will be supplimental to cudnn.
package layers

import (
	"errors"
	"fmt"
	"image"
	"strconv"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//IO is an all purpose struct that contains an x tensor and a dx tensor used for training
type IO struct {
	x    *tensor.Volume
	dx   *tensor.Volume
	dims []int32
}

func MakeJPG(folder, subfldr string, index int, img image.Image) error {
	return tensor.MakeJPG(folder, subfldr, index, img)
}
func (i *IO) SaveImagesToFile(dir string) error {
	x, dx, err := i.Images()
	if err != nil {
		return err
	}
	dirx := dir + "/x"
	err = saveimages(dirx, x)
	if err != nil {
		return err
	}
	dirdx := dir + "/dx"
	return saveimages(dirdx, dx)

}
func saveimages(folder string, imgs [][]image.Image) error {
	for i := 0; i < len(imgs); i++ {
		for k := 0; k < len(imgs[i]); k++ {
			err := MakeJPG(folder, "neuron"+strconv.Itoa(i)+"/Weight", k, imgs[i][k])
			if err != nil {
				return err
			}
		}
	}
	return nil
}
func (i *IO) Images() ([][]image.Image, [][]image.Image, error) {
	x, err := i.x.ToImages()
	if err != nil {
		return nil, nil, err
	}
	dx, err := i.dx.ToImages()
	if err != nil {
		return nil, nil, err
	}
	return x, dx, nil
}

//Properties returns the tensorformat, datatype and a slice of dims that describe the tensor
func (i *IO) Properties() (gocudnn.TensorFormat, gocudnn.DataType, []int32, error) {
	return i.x.Properties()
}

//DeltaT returns d tensor
func (i *IO) DeltaT() *tensor.Volume {
	return i.dx
}

//T returns the tensor
func (i *IO) T() *tensor.Volume {
	return i.x
}
func addtoerror(addition string, current error) error {
	errorstring := current.Error()
	return errors.New(addition + ": " + errorstring)
}
func TrainingInputIO(fmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
	inputdims, answerdims []int32,
	image, answer *gocudnn.GoPointer,
	managed bool,
) (*IO, error) {
	x, err := tensor.Build(fmt, dtype, inputdims, managed)
	if err != nil {
		x.Destroy()
		err = addtoerror("Building InputDims", err)
		return nil, err
	}
	dx, err := tensor.Build(fmt, dtype, answerdims, managed)
	if err != nil {
		err = addtoerror("Building answerdims", err)
		x.Destroy()
		dx.Destroy()
		return nil, err
	}
	err = x.LoadMem(image)
	if err != nil {
		err = addtoerror("Loading Images", err)
		return nil, err
	}
	err = dx.LoadMem(answer)
	if err != nil {
		err = addtoerror("Loading Answers", err)
		return nil, err
	}
	return &IO{
		x:  x,
		dx: dx,
	}, nil

}

//BuildIO builds an IO
func BuildIO(fmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32, managed bool) (*IO, error) {
	x, err := tensor.Build(fmt, dtype, dims, managed)
	if err != nil {
		x.Destroy()
		return nil, err
	}
	dx, err := tensor.Build(fmt, dtype, dims, managed)
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

//LoadTValues loads a piece of memory that was made in golang and loads into a tensor volume in cuda.
func (i *IO) LoadTValues(input *gocudnn.GoPointer) error {
	return i.x.LoadMem(input)
}

//LoadDeltaTValues loads a piece of memory that was made in golang and loads into a delta tensor volume in cuda.
func (i *IO) LoadDeltaTValues(input *gocudnn.GoPointer) error {
	return i.dx.LoadMem(input)
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
