package tensor

import (
	"io/ioutil"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/gocu"
)

//Info struct contains the info that is needed to build a volume
type Info struct {
	Format   gocudnn.TensorFormat `json:"Format,omitempty"`
	DataType gocudnn.DataType     `json:"DataType,omitempty"`
	Nan      gocudnn.NANProp      `json:"Nan,omitempty"`
	Dims     []int32              `json:"Dims,omitempty"`
	Stride   []int32              `json:"Stride,omitempty"`
	MaxDims  []int32              `json:"max_dims,omitempty"`
	Data     []byte               `json:"data,omitempty"`
}

//MakeInfo makes an info struct
func MakeInfo(frmt gocudnn.TensorFormat, dtype gocudnn.DataType, currnetdims, maxdims []int32) Info {
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
	rw := t.Malloced.NewReadWriter(nil)

	vals, err := ioutil.ReadAll(rw)
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

	var err error
	tens, err := gocudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	filts, err := gocudnn.CreateFilterDescriptor()
	if err != nil {
		return nil, err
	}
	err = tens.Set(i.Format, i.DataType, i.Dims, i.Stride)
	if err != nil {
		return nil, err
	}
	err = filts.Set(i.DataType, i.Format, i.Dims)
	size, err := tens.GetSizeInBytes()
	newmemer, err := nvidia.MallocGlobal(handle, size)
	if err != nil {
		return nil, err
	}

	vol := &Volume{
		Malloced: newmemer,
		current: &tensordescriptor{
			tD:      tens,
			fD:      filts,
			dims:    i.Dims,
			strides: utils.FindStridesInt32(i.Dims),
		},
		frmt:  i.Format,
		dtype: i.DataType,
	}
	if i.Data == nil {
		return vol, nil
	}
	goptr, err := gocu.MakeGoMem(i.Data)
	if err != nil {
		return nil, err
	}
	err = vol.LoadMem(handle, goptr, (uint)(len(i.Data)))
	if err != nil {
		return nil, err
	}
	return vol, nil
}
