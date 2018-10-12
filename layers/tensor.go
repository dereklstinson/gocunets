//Package layers contains shared things between layers.  It also contains functions that will be supplimental to cudnn.
package layers

import (
	"errors"
	"fmt"
	"image"
	"strconv"

	"github.com/dereklstinson/GoCuNets/cpu"
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//IO is an all purpose struct that contains an x tensor and a dx tensor used for training
type IO struct {
	x       *tensor.Volume
	dx      *tensor.Volume
	input   bool
	dims    []int32
	managed bool
}

//Settings contains the info that is needed to build an IO
type Settings struct {
	Dims     []int32                `json:"dims,omitempty"`
	Managed  bool                   `json:"managed,omitempty"`
	Format   gocudnn.TensorFormat   `json:"format,omitempty"`
	DataType gocudnn.DataType       `json:"data_type,omitempty"`
	NanProp  gocudnn.PropagationNAN `json:"nan_prop,omitempty"`
}

//Info is a struct that contains all the information to build an IO struct
type Info struct {
	NetworkInput bool        `json:"NetworkInput"`
	Dims         []int32     `json:"Dims"`
	Unified      bool        `json:"Unified"`
	X            tensor.Info `json:"X"`
	Dx           tensor.Info `json:"dX"`
}

//IsManaged returns if it is managed by cuda memory management system
func (i *IO) IsManaged() bool {
	return i.managed
}

//Info returns info
func (i *IO) Info() (Info, error) {
	x, err := i.x.Info()
	if err != nil {
		return Info{}, err
	}
	dx, err := i.dx.Info()
	if err != nil {
		return Info{}, err
	}
	return Info{
		NetworkInput: i.input,
		Dims:         i.dims,
		Unified:      i.managed,
		X:            x,
		Dx:           dx,
	}, nil
}

//IsInput returns if it is an input
func (i *IO) IsInput() bool {
	return i.input
}

//MakeJPG makes a jpg
func MakeJPG(folder, subfldr string, index int, img image.Image) error {
	return tensor.MakeJPG(folder, subfldr, index, img)
}

//SaveImagesToFile saves the images to file
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

//Images returns the images
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

//PlaceDeltaT will put a *tensor.Volume into the DeltaT place if and only if DeltaT is nil
func (i *IO) PlaceDeltaT(dT *tensor.Volume) {
	if i.dx != nil {
		i.dx.Destroy()
	}
	i.dx = dT

}

//PlaceT will put a *tensor.Volume into the T place if and only if T is nil
func (i *IO) PlaceT(T *tensor.Volume) {
	if i.x != nil {
		i.x.Destroy()
	}
	i.x = T
}

//ZeroClone Makes a zeroclone of the IO
func (i *IO) ZeroClone(handle *gocudnn.Handle) (*IO, error) {
	t, err := i.T().ZeroClone(handle)
	if err != nil {
		return nil, err
	}
	dt, err := i.T().ZeroClone(handle)
	if err != nil {
		t.Destroy()
		return nil, err
	}
	return &IO{
		x:       t,
		dx:      dt,
		dims:    i.dims,
		managed: i.managed,
	}, nil
}

//BuildIO builds a regular IO with both a T tensor and a DeltaT tensor
func BuildIO(frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32, managed bool) (*IO, error) {

	return buildIO(frmt, dtype, dims, managed, false)
}

//BuildNetworkInputIO builds an input IO which is an IO with DeltaT() set to nil. This is used for the input or the output of a network.
//If it is the output of a network in training. Then DeltaT will Need to be loaded with the labeles between batches.
func BuildNetworkInputIO(frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32, managed bool) (*IO, error) {
	return buildIO(frmt, dtype, dims, managed, true)
}

func buildIO(frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32, managed bool, input bool) (*IO, error) {

	if input == true {

		x, err := tensor.Build(frmt, dtype, dims, managed)
		if err != nil {
			x.Destroy()
			return nil, err
		}

		return &IO{

			x:       x,
			dx:      nil,
			input:   true,
			managed: managed,
		}, nil

	}
	x, err := tensor.Build(frmt, dtype, dims, managed)
	if err != nil {
		x.Destroy()
		return nil, err
	}
	dx, err := tensor.Build(frmt, dtype, dims, managed)
	if err != nil {
		x.Destroy()
		dx.Destroy()
		return nil, err
	}
	return &IO{
		x:       x,
		dx:      dx,
		managed: managed,
	}, nil
}

//LoadTValues loads a piece of memory that was made in golang and loads into an already created tensor volume in cuda.
func (i *IO) LoadTValues(input gocudnn.Memer) error {

	return i.x.LoadMem(input)
}

//
func (i *IO) GetLength() (int32, error) {
	_, _, dims, err := i.Properties()
	if err != nil {
		return 0, err
	}
	mult := int32(1)
	for i := 0; i < len(dims); i++ {
		mult *= dims[i]
	}
	return mult, nil
}

//MemIsManaged will return return if the memory is handled by cuda unified memory
func (i *IO) MemIsManaged() bool {
	return i.managed
}

//LoadDeltaTValues loads a piece of memory that was made in golang and loads into a previously created delta tensor volume in cuda.
func (i *IO) LoadDeltaTValues(input gocudnn.Memer) error {
	if i.input == true {
		return errors.New("Can't Load any values into DeltaT because it is exclusivly an Input")
	}
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

//ShapetoBatchIOcopyBWDCPU takes the batched IO that was created from the fwd process and replaces the delta Tensor values
func (i *IO) ShapetoBatchIOBWDCPU(batched *IO) error {
	var fmtflag gocudnn.TensorFormatFlag
	frmt, _, dimsy, err := batched.Properties()

	if err != nil {
		return err
	}
	_, _, dimsx, err := i.Properties()
	if err != nil {
		return err
	}
	blength, err := batched.GetLength()
	if err != nil {
		return err
	}
	ilength, err := i.GetLength()
	if err != nil {
		return err
	}

	dx := make([]float32, ilength)

	dy := make([]float32, blength)
	ptrdx, err := gocudnn.MakeGoPointer(dx)
	if err != nil {
		return err
	}
	ptrdy, err := gocudnn.MakeGoPointer(dy)
	if err != nil {
		return err
	}
	sizetdy, err := batched.DeltaT().Size()
	if err != nil {
		return err
	}
	if batched.managed == true {
		gocudnn.CudaMemCopy(ptrdy, batched.dx.Memer(), sizetdy, gocudnn.MemcpyKindFlag{}.Default())
	} else {
		gocudnn.CudaMemCopy(ptrdy, batched.dx.Memer(), sizetdy, gocudnn.MemcpyKindFlag{}.DeviceToHost())
	}

	if frmt == fmtflag.NCHW() {
		err = cpu.ShapeToBatchNCHW4DBackward(dx, dimsx, dy, dimsy)
		if err != nil {
			return err
		}
		return i.LoadDeltaTValues(ptrdx)

	} else if frmt == fmtflag.NHWC() {
		err = cpu.ShapeToBatchNHWC4DBackward(dx, dimsx, dy, dimsy)
		if err != nil {
			return err
		}
		return i.LoadDeltaTValues(ptrdx)
	}
	return errors.New("Unsupported Format")
}

//ShapetoBatchIOCopyCPU reshapes the makes a reshaped copy of the IO
func (i *IO) ShapetoBatchIOCopyFWDCPU(h, w int32) (*IO, error) {
	frmt, dtype, dims, err := i.Properties()
	if err != nil {
		return nil, err
	}
	length, err := i.GetLength()
	if err != nil {
		return nil, err
	}
	slice := make([]float32, length)
	i.T().Memer().FillSlice(slice)
	var fmtflag gocudnn.TensorFormatFlag
	if frmt == fmtflag.NCHW() {
		reshapedslice, rashapeddims, err := cpu.ShapeToBatchNCHW4DForward(slice, dims, h, w)
		if err != nil {
			return nil, err
		}
		newIO, err := BuildIO(frmt, dtype, rashapeddims, i.MemIsManaged())
		if err != nil {
			return nil, err
		}
		goptr, err := gocudnn.MakeGoPointer(reshapedslice)
		if err != nil {
			return nil, err
		}
		err = newIO.LoadTValues(goptr)
		return newIO, err
	}

	reshapedslice, rashapeddims, err := cpu.ShapeToBatchNHWC4DForward(slice, dims, h, w)
	if err != nil {
		return nil, err
	}
	newIO, err := BuildIO(frmt, dtype, rashapeddims, i.MemIsManaged())
	if err != nil {
		return nil, err
	}
	goptr, err := gocudnn.MakeGoPointer(reshapedslice)
	if err != nil {
		return nil, err
	}
	err = newIO.LoadTValues(goptr)
	return newIO, err
}

//ShapetoBatchIOCopyCPUWithSliceFloat32 reshapes the makes a reshaped copy of the IO
func (i *IO) ShapetoBatchIOCopyCPUWithSliceFloat32(h, w int32) (*IO, []float32, error) {
	frmt, dtype, dims, err := i.Properties()
	if err != nil {
		return nil, nil, err
	}
	length, err := i.GetLength()
	if err != nil {
		return nil, nil, err
	}
	slice := make([]float32, length)
	i.T().Memer().FillSlice(slice)
	var fmtflag gocudnn.TensorFormatFlag
	if frmt == fmtflag.NCHW() {
		reshapedslice, rashapeddims, err := cpu.ShapeToBatchNCHW4DForward(slice, dims, h, w)
		if err != nil {
			return nil, nil, err
		}
		newIO, err := BuildIO(frmt, dtype, rashapeddims, i.MemIsManaged())
		if err != nil {
			return nil, nil, err
		}
		goptr, err := gocudnn.MakeGoPointer(reshapedslice)
		if err != nil {
			return nil, nil, err
		}
		err = newIO.LoadTValues(goptr)
		return newIO, reshapedslice, err
	}

	reshapedslice, rashapeddims, err := cpu.ShapeToBatchNHWC4DForward(slice, dims, h, w)
	if err != nil {
		return nil, nil, err
	}
	newIO, err := BuildIO(frmt, dtype, rashapeddims, i.MemIsManaged())
	if err != nil {
		return nil, nil, err
	}
	goptr, err := gocudnn.MakeGoPointer(reshapedslice)
	if err != nil {
		return nil, nil, err
	}
	err = newIO.LoadTValues(goptr)
	return newIO, reshapedslice, err
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
