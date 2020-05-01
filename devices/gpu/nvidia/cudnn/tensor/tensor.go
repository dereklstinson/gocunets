//Package tensor is used to make tensors by using gocudnn.  It is currently not supporting what I call the "EX" functions.
//because the Tensor struct is also going to be carrying a filter descripter.  Also I call it "EX" functions loosly, because I think
//there is a miss labeling of the function names in cudnn. Basicly it is the set tensor fuctions that don't include the format and include
//the strides asdf
package tensor

//import "C"
import (
	"errors"
	"fmt"

	//"github.com/dereklstinson/cutil"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/utils"
	gocudnn "github.com/dereklstinson/gocudnn"
	"github.com/dereklstinson/gocudnn/gocu"
)

//Volume holds both a gocudnn.TensorD and gocudnn.FilterD and the allocated memory associated with it
type Volume struct {
	*nvidia.Malloced
	printvals bool
	//	dims      []int32
	current  *tensordescriptor
	op       tensops
	dtype    gocudnn.DataType
	propnan  gocudnn.NANProp
	frmt     gocudnn.TensorFormat
	min, max float32
	sizet    uint
	vol      int32
	ongpu    bool
}

//SetPropNan will change the default nan propigation flag from PropNanNon to PropNaN
func (t *Volume) SetPropNan() {
	t.propnan.Propigate()
}

//SetNotPropNan will set the nan propigation flag to NotPropigationNan (NotPropigationNan is default)
func (t *Volume) SetNotPropNan() {
	t.propnan.NotPropigate()

}

//BuildtoCudaHost stores the tensor memory to paged memory
func BuildtoCudaHost(handle *cudnn.Handler, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, startdims []int32) (*Volume, error) {

	current, err := maketensordescriptor(frmt, dtype, startdims)
	if err != nil {
		return nil, err
	}
	sizet, err := current.tD.GetSizeInBytes()
	if err != nil {
		return nil, err
	}
	vol := handle.FindVol(startdims)

	newmemer, err := nvidia.MallocHost(handle, sizet)
	if err != nil {
		return nil, err
	}
	return &Volume{
		Malloced: newmemer,
		current:  current,
		frmt:     frmt,
		dtype:    dtype,
		sizet:    sizet,
		vol:      vol,
		ongpu:    false,
	}, nil

}

//BuildEX builds a volume with settings passed.  And uses the mem passed as its memory. if mem is nil function will allocate memory to gpu itself
func BuildEX(handle *cudnn.Handler, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32, mem *nvidia.Malloced) (v *Volume, err error) {
	if mem == nil {
		return build(handle, frmt, dtype, dims)
	}
	sizet := gocudnn.FindSizeTfromVol(dims, dtype)
	if sizet > mem.SIB() {
		return nil, errors.New(" gocudnn.FindSizeTfromVol(dims, dtype)>mem.SIB()")
	}
	v = new(Volume)
	v.Malloced = mem
	vol := utils.FindVolumeInt32(dims, nil)
	v.current, err = maketensordescriptor(frmt, dtype, dims)
	v.sizet = sizet
	v.vol = vol

	return v, err
}

//Build creates a tensor and mallocs the memory for the tensor.
func Build(handle *cudnn.Handler, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32) (*Volume, error) {

	return build(handle, frmt, dtype, dims)
}

//BuildWeights builds the weights
func BuildWeights(handle *cudnn.Handler, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32) (*Volume, error) {
	return build(handle, frmt, dtype, dims)
}

//NormalRand sets the values in the volume to some Normalized Noise
func (t *Volume) NormalRand(h *cudnn.Handler, mean, std float32) error {
	rng := h.GetCuRNG()
	if rng == nil {
		return errors.New("Handlers rng is nil")

	}
	return rng.NormalFloat32(t, t.sizet, mean, std)

}

/*
func BuildRandNorm(handle *cudnn.Handler, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, groupcount int32,currentdims []int32, mean, std float32, seed uint64, static bool) (*Volume, error) {
	vol, err := build(handle, frmt, dtype, currentdims, nil, static)
	if err != nil {
		return nil, err
	}
	var x curand.RngType
	vol.randgen = curand.CreateGenerator((x.PseudoDefault()))

	err = vol.randgen.SetPsuedoSeed(seed)
	if err != nil {
		return nil, err
	}

	err = vol.randgen.SetStream(handle.Stream())
	if err != nil {
		return nil, err
	}
	err = vol.randgen.NormalFloat32(vol, vol.sizet, mean, std)
	//vol
	return vol, err

}
*/

//BuildRandNorm sets a randomnorm volume that can have its values set to random values over and over again
func BuildRandNorm(handle *cudnn.Handler, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, currentdims []int32, mean, std float32) (*Volume, error) {
	vol, err := build(handle, frmt, dtype, currentdims)
	if err != nil {
		return nil, err
	}

	err = handle.GetCuRNG().NormalFloat32(vol, vol.sizet, mean, std)
	//vol
	return vol, err
}

//Build creates a tensor and mallocs the memory for the tensor
func build(handle *cudnn.Handler, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32) (*Volume, error) {
	current, err := maketensordescriptor(frmt, dtype, dims)
	if err != nil {
		return nil, err
	}
	sizet := gocudnn.FindSizeTfromVol(dims, dtype)
	newmemer, err := nvidia.MallocGlobal(handle, sizet)
	if err != nil {
		return nil, err
	}
	return &Volume{
		Malloced: newmemer,
		current:  current,
		frmt:     frmt,
		dtype:    dtype,

		sizet: (sizet),
		vol:   utils.FindVolumeInt32(dims, nil),
		ongpu: true,
	}, nil

}

//TogglePrintValueForStringer is for debugging.  It will include the array values in String().
func (t *Volume) TogglePrintValueForStringer() {
	if !t.printvals {
		t.printvals = true
	} else {
		t.printvals = false
	}
	return
}
func (t *Volume) String() string {

	if t.printvals {
		return fmt.Sprintf("Volume{\n%v\nValues: %v\n}\n", t.current.tD.String(), t.stringmem())
	}
	return fmt.Sprintf("Volume{\n%v\nValues: %v\n}\n", t.current.tD.String(), "Use (t *Volume) TogglePrintValueForStringer() to print values")

	/*	flg := t.frmt
		switch t.frmt {

		case flg.NCHW():

		case flg.NHWC():

		}
	*/
}

//DataType returns the datatype of the volume
func (t *Volume) DataType() gocudnn.DataType {
	return t.dtype
}

//Format returns the format of the volume
func (t *Volume) Format() gocudnn.TensorFormat {
	return t.frmt
}

//TDStrided is a function that returns the strided tensor descriptor.
func (t *Volume) TDStrided() *gocudnn.TensorD {
	return t.current.tDstrided
}

//TD returns the tensor descriptor for Tensor
func (t *Volume) TD() *gocudnn.TensorD {
	if t == nil {
		panic("TD t is nill")
	}
	return t.current.tD
}

//FD returns the filter descriptor for Tensor
func (t *Volume) FD() *gocudnn.FilterD {
	if t == nil {
		panic("TD t is nill")
	}

	return t.current.fD
}

//Dims returns the dims of the tensor
func (t *Volume) Dims() []int32 {
	if t.current.dims == nil {

		t.current.dims = t.TD().Dims()
	}

	return t.current.dims
}

//Vol returns the volume of tensor
func (t *Volume) Vol() int32 {
	return t.vol
}

//SIB will returns the size in bytes
func (t *Volume) SIB() uint {
	return t.sizet
}

/*
//Ptr satisfies the cutil.Mem interface
func (t *Volume) Ptr() unsafe.Pointer {
	if t == nil {
		return nil
	} else if t.Malloced == nil {
		return nil
	}

	return t
}
*/

/*
//DPtr satisfies the cutil.Mem interface
func (t *Volume) DPtr() *unsafe.Pointer {
	return t.DPtr()
}

//Memer returns the Memer for Tensor
func (t *Volume) Memer() *nvidia.Malloced {
	return t
}
*/

//Properties returns the properties of the tensor
func (t *Volume) Properties() (gocudnn.TensorFormat, gocudnn.DataType, []int32, error) {
	frmt, a, b, _, err := t.current.tD.Get()
	return frmt, a, b, err

}

/*
//FillSlice will fill the slice
func (t *Volume) FillSlice(input interface{}, length int32) error {
	if utils.FindVolumeInt32(t.current.dims, nil) != length {
		return errors.New("Volume Doesn't Match Length of input")
	}
	return t.FillSlice(input)
}
*/

//ZeroClone clones the Volume with only zeros
func ZeroClone(h *cudnn.Handler, t *Volume) (*Volume, error) {

	if t.current.tD == nil || t.current.fD == nil || t == nil {
		return nil, errors.New("Tensor is nil")
	}
	frmt, dtype, dims, err := t.Properties()
	if err != nil {
		return nil, err
	}

	current, err := maketensordescriptor(frmt, dtype, dims)
	if err != nil {
		return nil, err
	}

	if t.ongpu {
		newmemer, err := nvidia.MallocGlobal(h, t.sizet)
		if err != nil {
			return nil, err
		}
		return &Volume{
			current:  current,
			frmt:     frmt,
			dtype:    dtype,
			Malloced: newmemer,
			sizet:    t.sizet,
			vol:      t.vol,
			ongpu:    t.ongpu,
		}, nil
	}
	newmemer, err := nvidia.MallocHost(h, t.sizet)
	if err != nil {
		return nil, err
	}
	return &Volume{
		current:  current,
		frmt:     frmt,
		dtype:    dtype,
		Malloced: newmemer,
		sizet:    t.sizet,
		vol:      t.vol,
		ongpu:    t.ongpu,
	}, nil
}

//arraysize will return the size of the array and will return 0 if unsupported type is used.
func arraysize(dtype gocudnn.DataType, size uint) int {
	var flg gocudnn.DataType
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
func (t *Volume) nhwctensorstringformated(data []float32) string {
	dims := t.Dims()
	strides := utils.FindStridesInt32(dims)
	var s string
	s = "\n"
	for i := int32(0); i < dims[0]; i++ {
		s = s + fmt.Sprintf("Batch[%v]{\n", i)
		for j := int32(0); j < dims[1]; j++ {
			//s = s + fmt.Sprintf("(h,w:%v,", j)
			for k := int32(0); k < dims[2]; k++ {
				s = s + fmt.Sprintf("(%v,%v)[ ", j, k)
				for l := int32(0); l < dims[3]; l++ {
					val := data[i*strides[0]+j*strides[1]+k*strides[2]+l*strides[3]]
					if val >= 0 {
						s = s + fmt.Sprintf(" %.5f ", val)
					} else {
						s = s + fmt.Sprintf("%.5f ", val)
					}

				}
				s = s + "], "

			}
			s = s + "\n"
		}
		s = s + "}\n"
	}
	return s
}
func (t *Volume) nchwtensorstringformated(data []float32) string {
	//flg := t.Format()

	dims := t.Dims()
	strides := utils.FindStridesInt32(dims)
	var s string
	s = "\n"
	for i := int32(0); i < dims[0]; i++ {
		s = s + fmt.Sprintf("Batch[%v]{\n", i)
		for j := int32(0); j < dims[1]; j++ {
			s = s + fmt.Sprintf("\tChannel[%v]{\n", j)
			for k := int32(0); k < dims[2]; k++ {
				s = s + "\t\t"
				for l := int32(0); l < dims[3]; l++ {
					val := data[i*strides[0]+j*strides[1]+k*strides[2]+l*strides[3]]
					if val >= 0 {
						s = s + fmt.Sprintf(" %.4f ", val)
					} else {
						s = s + fmt.Sprintf("%.4f ", val)
					}

				}
				s = s + "\n"

			}
			s = s + "\t}\n"
		}
		s = s + "}\n"
	}
	return s
}
func (t *Volume) stringmem() string {
	var flg gocudnn.DataType
	var sib uint

	var as int
	var max = true
	if max {
		sib = t.sizet
		as = arraysize(t.dtype, sib)
	} else {
		sib = gocudnn.FindSizeTfromVol(t.current.dims, t.dtype)
		as = int(utils.FindVolumeInt32(t.current.dims, t.current.dims))
	}

	switch t.dtype {
	case flg.Double():

		array := make([]float64, as)
		ptr, err := gocu.MakeGoMem(array)
		if err != nil {
			return "[Error In MakeGoMem]"
		}
		err = nvidia.Memcpy(ptr, t, sib)
		if err != nil {
			return "[Error In Memcpy]"
		}

		return fmt.Sprintf("%v", array)

	case flg.Float():

		array := make([]float32, as)
		ptr, err := gocu.MakeGoMem(array)
		if err != nil {
			return "[Error In MakeGoMem]"
		}
		err = nvidia.Memcpy(ptr, t, sib)
		if err != nil {
			return "[Error In Memcpy]"
		}
		fmtflg := t.frmt
		if t.frmt == fmtflg.NCHW() {
			return t.nchwtensorstringformated(array)
		}
		if t.frmt == fmtflg.NHWC() {
			return t.nhwctensorstringformated(array)
		}
		return fmt.Sprint(array)

	//	return fmt.Sprintf("%v", array)
	case flg.Int32():

		array := make([]int32, as)
		ptr, err := gocu.MakeGoMem(array)
		if err != nil {
			return "[Error In MakeGoMem]"
		}
		err = nvidia.Memcpy(ptr, t, sib)
		if err != nil {
			return "[Error In Memcpy]"
		}
		return fmt.Sprintf("%v", array)
	case flg.UInt8():

		array := make([]byte, as)
		ptr, err := gocu.MakeGoMem(array)
		if err != nil {
			return "[Error In MakeGoMem]"
		}
		err = nvidia.Memcpy(ptr, t, sib)
		if err != nil {
			return "[Error In Memcpy]"
		}
		return fmt.Sprintf("%v", array)
	case flg.Int8():

		array := make([]int8, as)
		ptr, err := gocu.MakeGoMem(array)
		if err != nil {
			return "[Error In MakeGoMem]"
		}
		err = nvidia.Memcpy(ptr, t, sib)
		if err != nil {
			return "[Error In Memcpy]"
		}
		return fmt.Sprintf("%v", array)
	default:
		return "[Unsupported Format]"
	}

}
