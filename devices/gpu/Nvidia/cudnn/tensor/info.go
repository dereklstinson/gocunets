package tensor

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/Nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Info struct contains the info that is needed to build a volume
type Info struct {
	Format   cudnn.TensorFormat `json:"Format,omitempty"`
	DataType cudnn.DataType     `json:"DataType,omitempty"`
	Nan      cudnn.NanMode      `json:"Nan,omitempty"`
	Dims     []int32            `json:"Dims,omitempty"`
	MaxDims  []int32            `json:"max_dims,omitempty"`
	Values   []float64          `json:"Values,omitempty"`
}

//MakeInfo makes an info struct
func MakeInfo(frmt cudnn.TensorFormat, dtype cudnn.DataType, currnetdims, maxdims []int32) Info {
	return Info{
		Format:   frmt,
		DataType: dtype,
		Dims:     currnetdims,
		MaxDims:  maxdims,
	}
}

//Info returns an Info struct that is used for saving info. If an error is returned then the values of Info will be set to default golang's default
func (t *Volume) Info() (Info, error) {
	frmt, dtype, dims, err := t.Properties()

	if err != nil {
		return Info{}, err
	}
	dflgs := t.thelp.Flgs.Data

	size := utils.FindVolumeInt32(dims, nil)

	//I don't like this switch type stuff.  I am probably going to make something in the gocudnn package to get rid of this. I just haven't thought of a really easy way to implement this.
	vals := make([]float64, size)
	switch dtype.Cu() {
	case dflgs.Double():

		values := make([]float64, size)
		err = t.memgpu.FillSlice(values)
		if err != nil {
			return Info{}, err
		}
		for i := range values {
			vals[i] = float64(values[i])
		}
	case dflgs.Float():
		values := make([]float32, size)
		err = t.memgpu.FillSlice(values)
		if err != nil {
			return Info{}, err
		}
		for i := range values {
			vals[i] = float64(values[i])
		}
	case dflgs.Int32():
		values := make([]int32, size)
		err = t.memgpu.FillSlice(values)
		if err != nil {
			return Info{}, err
		}
		for i := range values {
			vals[i] = float64(values[i])
		}
	case dflgs.Int8():
		values := make([]float64, size)
		err = t.memgpu.FillSlice(values)
		if err != nil {
			return Info{}, err
		}
		for i := range values {
			vals[i] = float64(values[i])
		}

	default:
		return Info{}, errors.New("Unsupported Format : Most likely internal error. Contact Code Writer")
	}

	return Info{
		Format:   frmt,
		DataType: dtype,
		Dims:     dims,
		Values:   vals,
	}, nil
}

//Build is a method for Info that will retrun a volume type. If Weights is nil the memory will still be malloced on the cuda side.  So make sure to add values if needed.
func (i Info) Build(handle *cudnn.Handler) (*Volume, error) {
	var thelper gocudnn.Tensor
	var fhelper gocudnn.Filter
	if len(i.Dims) < 4 {
		return nil, errors.New("Dims less than 4. Create A 4 dim Tensor and set dims not needed to 1")
	}
	var newmemer *gocudnn.Malloced
	var tens *gocudnn.TensorD
	var filts *gocudnn.FilterD
	var err error
	if len(i.Dims) > 4 {
		tens, err = thelper.NewTensorNdDescriptorEx(i.Format.Cu(), i.DataType.Cu(), i.Dims)
		if err != nil {
			return nil, err
		}
		filts, err = fhelper.NewFilterNdDescriptor(i.DataType.Cu(), i.Format.Cu(), i.Dims)
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
		if handle.Unified() == true {
			newmemer, err = gocudnn.MallocManaged(size, gocudnn.ManagedMemFlag{}.Global())
			if err != nil {

				tens.DestroyDescriptor()
				filts.DestroyDescriptor()
				return nil, err
			}
			newmemer.Set(0)

		} else {

			newmemer, err = gocudnn.Malloc(size)
			if err != nil {

				tens.DestroyDescriptor()
				filts.DestroyDescriptor()
				return nil, err

			}
			newmemer.Set(0)

		}

	} else {

		tens, err = thelper.NewTensor4dDescriptor(i.DataType.Cu(), i.Format.Cu(), i.Dims)
		if err != nil {
			return nil, err
		}
		filts, err = fhelper.NewFilter4dDescriptor(i.DataType.Cu(), i.Format.Cu(), i.Dims)
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
		if handle.Unified() == true {

			newmemer, err = gocudnn.MallocManaged(size, gocudnn.ManagedMemFlag{}.Global())
			if err != nil {

				tens.DestroyDescriptor()
				filts.DestroyDescriptor()
				return nil, err

			}
			newmemer.Set(0)
		} else {
			newmemer, err = gocudnn.Malloc(size)
			if err != nil {

				tens.DestroyDescriptor()
				filts.DestroyDescriptor()
				return nil, err

			}
			newmemer.Set(0)

		}

	}

	vol := &Volume{
		current: &tensordescriptor{
			tD:      tens,
			fD:      filts,
			dims:    i.Dims,
			strides: utils.FindStridesInt32(i.Dims),
		},
		frmt:   i.Format,
		dtype:  i.DataType,
		memgpu: newmemer,
	}
	if i.Values == nil {
		return vol, nil
	}
	goptr, err := gocudnn.MakeGoPointer(i.Values)
	if err != nil {
		return nil, err
	}
	err = vol.LoadMem(handle, goptr, cudnn.SizeT(goptr.ByteSize()))
	if err != nil {
		return nil, err
	}
	return vol, nil
}
