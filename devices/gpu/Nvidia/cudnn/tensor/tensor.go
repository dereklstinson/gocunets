//Package tensor is used to make tensors by using gocudnn.  It is currently not supporting what I call the "EX" functions.
//because the Tensor struct is also going to be carrying a filter descripter.  Also I call it "EX" functions loosly, because I think
//there is a miss labeling of the function names in cudnn. Basicly it is the set tensor fuctions that don't include the format and include
//the strides asdf
package tensor

//import "C"
import (
	"errors"
	"fmt"

	"github.com/dereklstinson/GoCuNets/devices/gpu/Nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/curand"
	"github.com/dereklstinson/GoCudnn/gocu"
)

//Volume holds both a gocudnn.TensorD and gocudnn.FilterD and the allocated memory associated with it
type Volume struct {
	current  *tensordescriptor
	previous []*tensordescriptor
	memgpu   gocu.Mem
	thelp    gocudnn.Tensor
	fhelp    gocudnn.Filter
	ophelp   gocudnn.OpTensor
	randgen  *curand.Generator
	op       tensops
	dtype    cudnn.DataType
	propnan  cudnn.NanMode
	frmt     cudnn.TensorFormat
	min, max float32
	maxsizet uint
	maxvol   int32
	ongpu    bool
	weights  bool
	//	ongpu    bool

	//scalar gocudnn.CScalar
}

//SetPropNan will change the default nan propigation flag from PropNanNon to PropNaN
func (t *Volume) SetPropNan() {
	t.propnan = cudnn.NanMode(t.thelp.Flgs.NaN.PropagateNan())
}

//SetNotPropNan will set the nan propigation flag to NotPropigationNan (NotPropigationNan is default)
func (t *Volume) SetNotPropNan() {
	t.propnan = cudnn.NanMode(t.thelp.Flgs.NaN.NotPropagateNan())

}

//Flags returns a struct that passes gocudnn flags through methods used in building the tensor
func Flags() gocudnn.TensorFlags {
	return gocudnn.TensorFlags{}
}

//ChangeDims will change the dims of the volume. As long as they are within the max volume of the maxdims size.
// There is also a fifo queue that this will check first of size 10 that this will check first, and move it to the front.
func (t *Volume) ChangeDims(dims []int32) error {
	if (utils.FindVolumeInt32(dims, nil)) > t.maxvol {
		return errors.New("volume of dims passed are larger than the max volume for this Volume")
	}
	var err error
	for i := range t.previous {
		if utils.CompareInt32(dims, t.previous[i].dims) {
			t.current = t.previous[i]
			firstinfirstout(t.current, t.previous[:i+1])
			return nil
		}
	}
	t.current, err = maketensordescriptor(t.frmt, t.dtype, dims)
	if err != nil {
		return err
	}
	firstinfirstout(t.current, t.previous)
	return nil
}

//BuildtoCudaHost stores the tensor memory to paged memory
func BuildtoCudaHost(handle *cudnn.Handler, frmt cudnn.TensorFormat, dtype cudnn.DataType, startdims []int32) (*Volume, error) {
	previous := make([]*tensordescriptor, 10)
	current, err := maketensordescriptor(frmt, dtype, startdims)
	if err != nil {
		return nil, err
	}

	previous[0] = current
	sizet := handle.FindMaxSizeT(startdims)
	maxvol := handle.FindMaxVol(startdims)
	if handle.Unified() {
		newmemer, err := gocudnn.MallocManaged(sizet.Cu(), gocudnn.ManagedMemFlag{}.Host())
		if err != nil {
			return nil, err
		}
		return &Volume{
			current:  current,
			previous: previous,
			frmt:     frmt,
			dtype:    dtype,
			memgpu:   newmemer,
			maxsizet: sizet,
			maxvol:   maxvol,
			ongpu:    false,
		}, nil
	}
	newmemer, err := gocudnn.MallocHost(sizet.Cu())
	if err != nil {
		return nil, err
	}
	return &Volume{
		current:  current,
		previous: previous,
		frmt:     frmt,
		dtype:    dtype,
		memgpu:   newmemer,
		maxsizet: sizet,
		maxvol:   maxvol,
		ongpu:    false,
	}, nil

}

//BuildWithMaxDims is used if you already know the max dims.  This would be good on using something like S2B or B2S
func BuildWithMaxDims(handle *cudnn.Handler, frmt cudnn.TensorFormat, dtype cudnn.DataType, currentdims, maxdims []int32) (*Volume, error) {
	return build(handle, frmt, dtype, currentdims, maxdims, false)
}

//Build creates a tensor and mallocs the memory for the tensor. Max dims will change the amount of memory allocated to it.
//  Technically, these are not the max dims. As long as the new dim volume is <= max dim volume.
//If maxvol is zero or negative it will chose the max volume to be the size of the currentdims
func Build(handle *cudnn.Handler, frmt cudnn.TensorFormat, dtype cudnn.DataType, currentdims []int32) (*Volume, error) {

	return build(handle, frmt, dtype, currentdims, nil, false)
}

//BuildWeights builds the weights
func BuildWeights(handle *cudnn.Handler, frmt cudnn.TensorFormat, dtype cudnn.DataType, currentdims []int32) (*Volume, error) {
	return build(handle, frmt, dtype, currentdims, nil, true)
}

//AddRandNormalGenerator will add a random normal generator to the volume
func (t *Volume) AddRandNormalGenerator(handle *cudnn.Handler, seed uint64) error {
	t.randgen = gocudnn.CreateCuRandGenerator((gocudnn.CuRandRngTypeFlag{}.PseudoDefault()))
	err := t.randgen.SetPsuedoSeed(seed)
	if err != nil {
		return err
	}
	stream, err := handle.Cudnn().GetStream()
	if err != nil {
		return err
	}
	if stream != nil {
		return t.randgen.SetStream(stream)
	}
	fmt.Println("No Stream in handle")
	return nil

}

//NormalRand sets the values in the volume to some Normalized Noise
func (t *Volume) NormalRand(mean, std float32) error {
	if t.randgen == nil {
		return errors.New("Need to build a random volume using BuildRandNorm")

	}
	//The mem might need to be set to zero. Maybe
	return t.randgen.NormalFloat32(t.memgpu, mean, std)
}

//BuildRandNorm sets a randomnorm volume that can have its values set to random values over and over again
func BuildRandNorm(handle *cudnn.Handler, frmt cudnn.TensorFormat, dtype cudnn.DataType, currentdims []int32, mean, std float32, seed uint64, static bool) (*Volume, error) {
	vol, err := build(handle, frmt, dtype, currentdims, nil, static)
	if err != nil {
		return nil, err
	}

	vol.randgen = gocudnn.CreateCuRandGenerator(gocudnn.CuRandRngTypeFlag{}.PseudoDefault())

	err = vol.randgen.SetPsuedoSeed(seed)
	if err != nil {
		return nil, err
	}

	err = vol.randgen.SetStream(handle.Stream())
	if err != nil {
		return nil, err
	}
	err = vol.randgen.NormalFloat32(vol.memgpu, mean, std)
	//vol.memgpu
	return vol, err
}
func firstinfirstout(new *tensordescriptor, previous []*tensordescriptor) {

	for i := len(previous) - 1; i > 0; i-- {
		previous[i] = previous[i-1]

	}
	previous[0] = new

}

//Build creates a tensor and mallocs the memory for the tensor
func build(handle *cudnn.Handler, frmt cudnn.TensorFormat, dtype cudnn.DataType, startdims, maxdims []int32, weights bool) (*Volume, error) {
	switch weights {
	case true:
		previous := make([]*tensordescriptor, 10)
		current, err := maketensordescriptor(frmt, dtype, startdims)
		if err != nil {
			return nil, err
		}

		previous[0] = current

		sizet := gocudnn.FindSizeTfromVol(startdims, dtype.Cu())
		if handle.Unified() {
			newmemer, err := gocudnn.MallocManaged(sizet, gocudnn.ManagedMemFlag{}.Global())
			if err != nil {
				return nil, err
			}
			return &Volume{
				current:  current,
				previous: previous,
				frmt:     frmt,
				dtype:    dtype,
				memgpu:   newmemer,
				maxsizet: cudnn.SizeT(sizet),
				maxvol:   utils.FindVolumeInt32(startdims, nil),
				ongpu:    true,
				weights:  weights,
			}, nil
		}
		newmemer, err := gocudnn.Malloc(sizet)
		if err != nil {
			return nil, err
		}
		return &Volume{
			current:  current,
			previous: previous,
			frmt:     frmt,
			dtype:    dtype,
			memgpu:   newmemer,
			maxsizet: cudnn.SizeT(sizet),
			maxvol:   utils.FindVolumeInt32(startdims, nil),
			ongpu:    true,
			weights:  weights,
		}, nil
	case false:
		previous := make([]*tensordescriptor, 10)
		current, err := maketensordescriptor(frmt, dtype, startdims)
		if err != nil {
			return nil, err
		}

		previous[0] = current
		var maxvol int32
		var sizet cudnn.SizeT
		if maxdims == nil {
			sizet = handle.FindMaxSizeT(startdims)
			maxvol = handle.FindMaxVol(startdims)
		} else {
			maxvol = utils.FindVolumeInt32(maxdims, nil)
			sizet = cudnn.SizeT(gocudnn.FindSizeTfromVol(maxdims, dtype.Cu()))
		}

		if handle.Unified() {
			newmemer, err := gocudnn.MallocManaged(sizet.Cu(), gocudnn.ManagedMemFlag{}.Global())
			if err != nil {
				return nil, err
			}
			return &Volume{
				current:  current,
				previous: previous,
				frmt:     frmt,
				dtype:    dtype,
				memgpu:   newmemer,
				maxsizet: sizet,
				maxvol:   maxvol,
				ongpu:    true,
				weights:  weights,
			}, nil
		}
		newmemer, err := gocudnn.Malloc(sizet.Cu())
		if err != nil {
			return nil, err
		}
		return &Volume{
			current:  current,
			previous: previous,
			frmt:     frmt,
			dtype:    dtype,
			memgpu:   newmemer,
			maxsizet: sizet,
			maxvol:   maxvol,
			ongpu:    true,
			weights:  weights,
		}, nil
	}
	previous := make([]*tensordescriptor, 10)
	current, err := maketensordescriptor(frmt, dtype, startdims)
	if err != nil {
		return nil, err
	}
	previous[0] = current
	var maxvol int32
	var sizet cudnn.SizeT
	if maxdims == nil {
		sizet = handle.FindMaxSizeT(startdims)
		maxvol = handle.FindMaxVol(startdims)
	} else {
		maxvol = utils.FindVolumeInt32(maxdims, nil)
		sizet = cudnn.SizeT(gocudnn.FindSizeTfromVol(maxdims, dtype.Cu()))
	}

	if handle.Unified() {
		newmemer, err := gocudnn.MallocManaged(sizet.Cu(), gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			return nil, err
		}
		return &Volume{
			current:  current,
			previous: previous,
			frmt:     frmt,
			dtype:    dtype,
			memgpu:   newmemer,
			maxsizet: sizet,
			maxvol:   maxvol,
			ongpu:    true,
			weights:  weights,
		}, nil
	}
	newmemer, err := gocudnn.Malloc(sizet.Cu())
	if err != nil {
		return nil, err
	}
	return &Volume{
		current:  current,
		previous: previous,
		frmt:     frmt,
		dtype:    dtype,
		memgpu:   newmemer,
		maxsizet: sizet,
		maxvol:   maxvol,
		ongpu:    true,
		weights:  weights,
	}, nil

}

//DataType returns the datatype of the volume
func (t *Volume) DataType() cudnn.DataType {
	return t.dtype
}

//Format returns the format of the volume
func (t *Volume) Format() cudnn.TensorFormat {
	return t.frmt
}

//TDStrided is a function that returns the strided tensor descriptor.
func (t *Volume) TDStrided() *gocudnn.TensorD {
	return t.current.tDstrided
}

//TD returns the tensor descriptor for Tensor
func (t *Volume) TD() *gocudnn.TensorD {
	return t.current.tD
}

//FD returns the filter descriptor for Tensor
func (t *Volume) FD() *gocudnn.FilterD {
	return t.current.fD
}

//Dims returns the dims of the tensor
func (t *Volume) Dims() []int32 {
	return t.current.dims
}

//MaxVol Returns the MaxVol that the tensor can be built to given the memory saved to it
func (t *Volume) MaxVol() int32 {
	return t.maxvol
}

//MaxSizeT will return the max size in bytes that this tensor can build with
func (t *Volume) MaxSizeT() cudnn.SizeT {
	return t.maxsizet
}

//Memer returns the Memer for Tensor
func (t *Volume) Memer() gocu.Mem {
	return t.memgpu
}

//CurrentSizeT returns the size in bytes for the current tensor
func (t *Volume) CurrentSizeT() cudnn.SizeT {
	return cudnn.SizeT(gocudnn.FindSizeTfromVol(t.current.dims, t.dtype.Cu()))
}

//Properties returns the properties of the tensor
func (t *Volume) Properties() (cudnn.TensorFormat, cudnn.DataType, []int32, error) {
	a, b, _, err := t.current.tD.GetDescrptor()
	return t.frmt, cudnn.DataType(a), b, err

}

//FillSlice will fill the slice
func (t *Volume) FillSlice(input interface{}, length int32) error {
	if utils.FindVolumeInt32(t.current.dims, nil) != length {
		return errors.New("Slice Doesn't Match Length of input")
	}
	return t.memgpu.FillSlice(input)
}

//ZeroClone returns a zero clone of the the memory
func (t *Volume) ZeroClone(handle *cudnn.Handler) (*Volume, error) {

	if t.current.tD == nil || t.current.fD == nil || t.memgpu == nil {
		return nil, errors.New("Tensor is nil")
	}
	frmt, dtype, dims, err := t.Properties()
	if err != nil {
		return nil, err
	}
	previous := make([]*tensordescriptor, 10)
	current, err := maketensordescriptor(frmt, dtype, dims)
	if err != nil {
		return nil, err
	}

	previous[0] = current

	if handle.Unified() {
		newmemer, err := gocudnn.MallocManaged(t.maxsizet.Cu(), gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			return nil, err
		}
		return &Volume{
			current:  current,
			previous: previous,
			frmt:     frmt,
			dtype:    dtype,
			memgpu:   newmemer,
			maxsizet: t.maxsizet,
			maxvol:   t.maxvol,
			ongpu:    t.ongpu,
			weights:  t.weights,
		}, nil
	}
	newmemer, err := gocudnn.Malloc(t.maxsizet.Cu())
	if err != nil {
		return nil, err
	}
	return &Volume{
		current:  current,
		previous: previous,
		frmt:     frmt,
		dtype:    dtype,
		memgpu:   newmemer,
		maxsizet: t.maxsizet,
		maxvol:   t.maxvol,
		ongpu:    t.ongpu,
		weights:  t.weights,
	}, nil

}

//arraysize will return the size of the array and will return 0 if unsupported type is used.
func arraysize(dtype cudnn.DataType, size gocudnn.SizeT) int {
	var flg gocudnn.DataTypeFlag
	x := int(size)
	switch dtype.Cu() {
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

func (t *Volume) printmem(comment string, kind gocudnn.MemcpyKind, max bool) error {
	var flg gocudnn.DataTypeFlag
	var sib gocudnn.SizeT

	var as int

	if max {
		sib = t.memgpu.ByteSize()
		as = arraysize(t.dtype, sib)
	} else {
		sib = gocudnn.FindSizeTfromVol(t.current.dims, t.dtype.Cu())
		as = int(utils.FindVolumeInt32(t.current.dims, t.current.dims))
	}

	switch t.dtype.Cu() {
	case flg.Double():

		array := make([]float64, as)
		ptr, err := gocudnn.MakeGoPointer(array)
		if err != nil {
			return err
		}
		err = gocudnn.CudaMemCopy(ptr, t.memgpu, sib, kind)
		if err != nil {
			return err
		}
		fmt.Printf("\n{")
		fmt.Println(comment, array)
		fmt.Printf("\n}")
	case flg.Float():

		array := make([]float32, as)
		ptr, err := gocudnn.MakeGoPointer(array)

		if err != nil {
			return err
		}
		err = gocudnn.CudaMemCopy(ptr, t.memgpu, sib, kind)
		if err != nil {
			return err
		}
		fmt.Printf("\n{")
		fmt.Println(comment, array)
		fmt.Printf("\n}")
	case flg.Int32():

		array := make([]int32, as)
		ptr, err := gocudnn.MakeGoPointer(array)

		if err != nil {
			return err
		}
		err = gocudnn.CudaMemCopy(ptr, t.memgpu, sib, kind)
		if err != nil {
			return err
		}
		fmt.Printf("\n{")
		fmt.Println(comment, array)
		fmt.Printf("\n}")
	case flg.UInt8():

		array := make([]byte, as)
		ptr, err := gocudnn.MakeGoPointer(array)

		if err != nil {
			return err
		}
		err = gocudnn.CudaMemCopy(ptr, t.memgpu, sib, kind)
		if err != nil {
			return err
		}
		fmt.Printf("\n{")
		fmt.Println(comment, array)
		fmt.Printf("\n}")
	case flg.Int8():

		array := make([]int8, as)
		ptr, err := gocudnn.MakeGoPointer(array)
		if err != nil {

			return err
		}
		err = gocudnn.CudaMemCopy(ptr, t.memgpu, sib, kind)
		if err != nil {

			return err
		}
		fmt.Printf("\n{")
		fmt.Println(comment, array)
		fmt.Printf("\n}")
	default:
		return errors.New("Unsupoorted Format")
	}

	return nil
}

//PrintCurrentTensor - Prints the memory that the current tensor is using
func (t *Volume) PrintCurrentTensor(handle *cudnn.Handler, comment string) error {
	var kflg gocudnn.MemcpyKindFlag
	if handle.Unified() {

		return t.printmem(comment, kflg.Default(), false)
	}
	if t.ongpu {
		return t.printmem(comment, kflg.DeviceToHost(), false)
	}
	return t.printmem(comment, kflg.HostToHost(), false)
}

//PrintMaxMem prints the unified Memory
func (t *Volume) PrintMaxMem(handle *cudnn.Handler, comment string) error {

	var kflg gocudnn.MemcpyKindFlag
	if handle.Unified() {
		return t.printmem(comment, kflg.Default(), true)
	}
	if t.ongpu {
		return t.printmem(comment, kflg.DeviceToHost(), true)
	}
	return t.printmem(comment, kflg.HostToHost(), true)
}
