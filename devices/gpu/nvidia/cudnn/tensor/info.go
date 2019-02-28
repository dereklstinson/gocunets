package tensor

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
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
	Data     []byte             `json:"data,omitempty"`
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

	vals := make([]byte, t.memgpu.TotalBytes())
	writen, err := t.memgpu.Write(vals)
	if err != nil {
		return Info{}, err
	}
	return Info{
		Format:   frmt,
		DataType: dtype,
		Dims:     dims,
		Data:     vals,
	}, nil
}

//Build is a method for Info that will retrun a volume type. If Weights is nil the memory will still be malloced on the cuda side.  So make sure to add values if needed.
func (i Info) Build(handle *cudnn.Handler) (*Volume, error) {
	var thelper gocudnn.Tensor
	var fhelper gocudnn.Filter
	if len(i.Dims) < 4 {
		return nil, errors.New("Dims less than 4. Create A 4 dim Tensor and set dims not needed to 1")
	}
	var newmemer *nvidia.Malloced
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

		newmemer, err = nvidia.MallocGlobal(handle,size)
		if err != nil {

			return nil, err
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
