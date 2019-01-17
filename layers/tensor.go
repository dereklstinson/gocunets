//Package layers contains shared things between layers.  It also contains functions that will be supplimental to cudnn.
package layers

import (
	"errors"
	"fmt"
	"sync"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//IO is an all purpose struct that contains an x tensor and a dx tensor used for training
type IO struct {
	x                                     *tensor.Volume
	dx                                    *tensor.Volume
	minx, maxx, avgx, norm1x, norm2x      *reduceop
	mindx, maxdx, avgdx, norm1dx, norm2dx *reduceop
	input                                 bool
	dims                                  []int32
	managed                               bool
	mux                                   sync.Mutex
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

//StoreDeltas will flip a flag to allow deltas to be stored on this IO.
//Useful when training gans when you don't want the errors when training the descriminator to propigate through this.
//You would want to switch it back when passing the errors for the generator.
func (i *IO) StoreDeltas(x bool) {
	i.input = x
}

//ClearDeltas allows the user to clear the deltas of the IO
func (i *IO) ClearDeltas() error {
	return i.dx.Memer().Set(0)

}

//MinX returns the minx value per batch in the tensor or if it is used for the filter it would be the minx value per neuron
func (i *IO) MinX(handle *cudnn.Handler) (x float32, e error) {
	i.mux.Lock()
	x, e = i.minx.Reduce(handle, i.T())
	i.mux.Unlock()
	return x, e

}

//MaxX returns the MaxX per batch value in the tensor or if it is used for the filter it would be the MaxX value per neuron
func (i *IO) MaxX(handle *cudnn.Handler) (x float32, e error) {
	i.mux.Lock()
	x, e = i.maxx.Reduce(handle, i.T())
	i.mux.Unlock()
	return x, e

}

//AvgX returns the Avg X value for the IO
func (i *IO) AvgX(handle *cudnn.Handler) (x float32, e error) {
	i.mux.Lock()
	x, e = i.avgx.Reduce(handle, i.T())
	i.mux.Unlock()
	return x, e

}

//Norm1X returns Norm1 X value for IO
func (i *IO) Norm1X(handle *cudnn.Handler) (x float32, e error) {
	i.mux.Lock()
	x, e = i.norm1x.Reduce(handle, i.T())
	i.mux.Unlock()
	return x, e

}

//Norm2X returns Norm2 X value for IO
func (i *IO) Norm2X(handle *cudnn.Handler) (x float32, e error) {
	i.mux.Lock()
	x, e = i.norm2x.Reduce(handle, i.T())
	i.mux.Unlock()
	return x, e

}

//MinDX returns the min dx value for th io.
func (i *IO) MinDX(handle *cudnn.Handler) (x float32, e error) {
	i.mux.Lock()
	x, e = i.mindx.Reduce(handle, i.T())
	i.mux.Unlock()
	return x, e

}

//MaxDX returns the Max DX  value for th io.
func (i *IO) MaxDX(handle *cudnn.Handler) (x float32, e error) {
	i.mux.Lock()
	x, e = i.maxdx.Reduce(handle, i.T())
	i.mux.Unlock()
	return x, e

}

//AvgDX returns the Avg DX value for the IO
func (i *IO) AvgDX(handle *cudnn.Handler) (x float32, e error) {
	i.mux.Lock()
	x, e = i.avgdx.Reduce(handle, i.T())
	i.mux.Unlock()
	return x, e

}

//Norm1DX returns Norm1 dX value for IO
func (i *IO) Norm1DX(handle *cudnn.Handler) (x float32, e error) {
	i.mux.Lock()
	x, e = i.norm1dx.Reduce(handle, i.T())
	i.mux.Unlock()
	return x, e

}

//Norm2DX returns Norm2 dX value for IO
func (i *IO) Norm2DX(handle *cudnn.Handler) (x float32, e error) {
	i.mux.Lock()
	x, e = i.norm2dx.Reduce(handle, i.T())
	i.mux.Unlock()
	return x, e

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

//CreateIOfromVolumes is a way to put a couple of volumes in there and have it fill the private properties of the IO.
// if dx is nil then the IO will be considered a network input tensor and the backward data will not propagate through this tensor
func CreateIOfromVolumes(x, dx *tensor.Volume) (*IO, error) {
	if x == nil {
		return nil, errors.New("createiofromvolumes x tensor.Volume can't be nil")
	}
	_, _, dims, err := x.Properties()
	if err != nil {
		return nil, err
	}
	var lcflg gocudnn.LocationFlag
	var managed bool
	if lcflg.Unified() == x.Memer().Stored() {
		managed = true

	}
	var isinput bool
	if dx == nil {
		isinput = true
	}
	return &IO{
		x:       x,
		dx:      dx,
		dims:    dims,
		managed: managed,
		input:   isinput,
	}, nil
}

func findslide(dims []int32) []int {
	multiplier := 1
	slide := make([]int, len(dims))
	for i := len(dims) - 1; i >= 0; i-- {
		slide[i] = multiplier
		multiplier *= int(dims[i])
	}
	return slide
}
func findvol(dims []int32) int {
	multiplier := 1

	for i := 0; i < len(dims); i++ {

		multiplier *= int(dims[i])
	}
	return multiplier
}

//Properties returns the tensorformat, datatype and a slice of dims that describe the tensor
func (i *IO) Properties() (cudnn.TensorFormat, cudnn.DataType, []int32, error) {
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

//SetXStatReducers will build the reducers for the IO x min,max,avg,norm1, norm2
func (i *IO) SetXStatReducers(handle *cudnn.Handler) (err error) {

	i.minx, err = buildminreduce(handle, i.T())
	if err != nil {
		return err
	}

	i.maxx, err = buildmaxreduce(handle, i.T())
	if err != nil {
		return err
	}
	i.avgx, err = buildavgreduce(handle, i.T())
	if err != nil {
		return err
	}
	i.norm1x, err = buildnorm1reduce(handle, i.T())
	if err != nil {
		return err
	}
	i.norm2x, err = buildnorm2reduce(handle, i.T())
	if err != nil {
		return err
	}
	return err
}

//SetDXStatReducers will build the reducers for the IO dx min,max,avg,norm1, norm2
func (i *IO) SetDXStatReducers(handle *cudnn.Handler) (err error) {
	i.mindx, err = buildminreduce(handle, i.DeltaT())
	if err != nil {
		return err
	}
	i.maxdx, err = buildmaxreduce(handle, i.DeltaT())
	if err != nil {
		return err
	}
	i.avgdx, err = buildavgreduce(handle, i.DeltaT())
	if err != nil {
		return err
	}
	i.norm1dx, err = buildnorm1reduce(handle, i.DeltaT())
	if err != nil {
		return err
	}
	i.norm2dx, err = buildnorm2reduce(handle, i.DeltaT())
	if err != nil {
		return err
	}
	return err
}

//PlaceDeltaT will put a *tensor.Volume into the DeltaT and destroy the previous memory held in the spot
func (i *IO) PlaceDeltaT(dT *tensor.Volume) {
	if i.dx != nil {
		i.dx.Destroy()
	}
	i.dx = dT

}

//PlaceT will put a *tensor.Volume into the T  and destroy the previous memory held in the spot
func (i *IO) PlaceT(T *tensor.Volume) {
	if i.x != nil {
		i.x.Destroy()
	}
	i.x = T
}

//ZeroClone Makes a zeroclone of the IO
func (i *IO) ZeroClone() (*IO, error) {
	frmt, dtype, dims, err := i.Properties()
	if err != nil {
		return nil, err
	}

	return BuildIO(frmt, dtype, dims, i.IsManaged())
}

//BuildIO builds a regular IO with both a T tensor and a DeltaT tensor
func BuildIO(frmt cudnn.TensorFormat, dtype cudnn.DataType, dims []int32, managed bool) (*IO, error) {

	return buildIO(frmt, dtype, dims, managed, false)
}

//BuildNormRandIO builds a regular IO with both a T tensor and a DeltaT tensor.  But the T tensor is randomized
func BuildNormRandIO(handle *cudnn.Handler, frmt cudnn.TensorFormat, dtype cudnn.DataType, dims []int32, mean, std float32, seed uint64, managed bool) (*IO, error) {
	return buildRandIO(handle, frmt, dtype, dims, mean, std, seed, managed, false)

}

//BuildNormRandInputIO builds a regular IO but the input is set to nil
func BuildNormRandInputIO(handle *cudnn.Handler, frmt cudnn.TensorFormat, dtype cudnn.DataType, dims []int32, mean, std float32, seed uint64, managed bool) (*IO, error) {
	return buildRandIO(handle, frmt, dtype, dims, mean, std, seed, managed, true)

}
func buildRandIO(handle *cudnn.Handler, frmt cudnn.TensorFormat, dtype cudnn.DataType, dims []int32, mean, std float32, seed uint64, managed bool, input bool) (*IO, error) {
	if input == true {

		x, err := tensor.BuildRandNorm(handle, frmt, dtype, dims, mean, std, seed, managed)
		if err != nil {
			return nil, err
		}

		return &IO{

			x:       x,
			dx:      nil,
			input:   true,
			managed: managed,
		}, nil

	}
	x, err := tensor.BuildRandNorm(handle, frmt, dtype, dims, mean, std, seed, managed)
	if err != nil {

		return nil, err
	}
	dx, err := tensor.Build(frmt, dtype, dims, managed)
	if err != nil {

		return nil, err
	}
	return &IO{
		x:       x,
		dx:      dx,
		managed: managed,
	}, nil

}

//BuildNetworkInputHost build the input tensor to paged memory on host ram
func BuildNetworkInputHost(frmt cudnn.TensorFormat, dtype cudnn.DataType, dims []int32, managed bool) (*IO, error) {
	x, err := tensor.BuildtoCudaHost(frmt, dtype, dims, managed)
	if err != nil {
		return nil, err
	}

	return &IO{
		x:       x,
		dx:      nil,
		input:   true,
		managed: managed,
	}, nil
}

//BuildNetworkInputIO builds an input IO which is an IO with DeltaT() set to nil. This is used for the input or the output of a network.
//If it is the output of a network in training. Then DeltaT will Need to be loaded with the labeles between batches.
func BuildNetworkInputIO(frmt cudnn.TensorFormat, dtype cudnn.DataType, dims []int32, managed bool) (*IO, error) {
	return buildIO(frmt, dtype, dims, managed, true)
}

//BuildNetworkOutputIOFromSlice will return IO with the slice put into the DeltaT() section of the IO
func BuildNetworkOutputIOFromSlice(frmt cudnn.TensorFormat, dtype cudnn.DataType, dims []int32, managed bool, slice []float32) (*IO, error) {

	chkr := int32(1)
	for i := 0; i < len(dims); i++ {
		chkr *= dims[i]
	}
	if chkr != int32(len(slice)) {
		return nil, errors.New("Slice passed length don't match dim volume")
	}
	sptr, err := gocudnn.MakeGoPointer(slice)
	if err != nil {
		return nil, err
	}
	newio, err := BuildIO(frmt, dtype, dims, managed)
	if err != nil {
		return nil, err
	}

	err = newio.LoadDeltaTValues(sptr)
	return newio, err
}

func buildIO(frmt cudnn.TensorFormat, dtype cudnn.DataType, dims []int32, managed bool, input bool) (*IO, error) {

	if input == true {

		x, err := tensor.Build(frmt, dtype, dims, managed)
		if err != nil {

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

		return nil, err
	}
	dx, err := tensor.Build(frmt, dtype, dims, managed)
	if err != nil {

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

//GetLength returns the length in int32
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
	var (
		err  error
		err1 error
	)
	if i.dx != nil {
		err = i.dx.Destroy()
		if err != nil {
			flag = true
		}
	}
	if i.x != nil {
		err1 = i.x.Destroy()
		if err1 != nil {
			flag = true
		}
	}

	if flag == true {
		return fmt.Errorf("error:x: %s,dx: %s", err, err1)
	}
	return nil
}
