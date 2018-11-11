package tensor

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Info struct contains the info that is needed to build a volume
type Info struct {
	Format   gocudnn.TensorFormat   `json:"Format"`
	DataType gocudnn.DataType       `json:"DataType"`
	Nan      gocudnn.PropagationNAN `json:"Nan"`
	Dims     []int32                `json:"Dims"`
	Unified  bool                   `json:"Unified"`
	Values   interface{}            `json:"Values"`
}

//MakeInfo makes an info struct
func MakeInfo(frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32, unified bool) Info {
	return Info{
		Format:   frmt,
		DataType: dtype,
		Dims:     dims,
		Unified:  unified,
	}
}

//Info returns an Info struct that is used for saving info. If an error is returned then the values of Info will be set to default golang's default
func (t *Volume) Info() (Info, error) {
	frmt, dtype, dims, err := t.Properties()

	if err != nil {
		return Info{}, err
	}
	dflgs := t.thelp.Flgs.Data
	var values interface{}
	size := utils.FindVolumeInt32(dims)

	//I don't like this switch type stuff.  I am probably going to make something in the gocudnn package to get rid of this. I just haven't thought of a really easy way to implement this.
	switch dtype.Cu() {
	case dflgs.Double():
		values = make([]float64, size)
	case dflgs.Float():
		values = make([]float32, size)
	case dflgs.Int32():
		values = make([]int32, size)
	case dflgs.Int8():
		values = make([]float64, size)

	default:
		return Info{}, errors.New("Unsupported Format : Most likely internal error. Contact Code Writer")
	}
	err = t.memgpu.FillSlice(values)
	if err != nil {
		return Info{}, err
	}
	return Info{
		Format:   frmt.Cu(),
		DataType: dtype.Cu(),
		Dims:     dims,
		Unified:  t.managed,
		Values:   values,
	}, nil
}

//Build is a method for Info that will retrun a volume type. If Weights is nil the memory will still be malloced on the cuda side.  So make sure to add values if needed.
func (i Info) Build() (*Volume, error) {
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
		tens, err = thelper.NewTensorNdDescriptorEx(i.Format, i.DataType, i.Dims)
		if err != nil {
			return nil, err
		}
		filts, err = fhelper.NewFilterNdDescriptor(i.DataType, i.Format, i.Dims)
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
		if i.Unified == true {
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

		tens, err = thelper.NewTensor4dDescriptor(i.DataType, i.Format, i.Dims)
		if err != nil {
			return nil, err
		}
		filts, err = fhelper.NewFilter4dDescriptor(i.DataType, i.Format, i.Dims)
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
		if i.Unified == true {

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
		tD:      tens,
		fD:      filts,
		memgpu:  newmemer,
		frmt:    i.Format,
		dtype:   i.DataType,
		dims:    i.Dims,
		strides: utils.FindStridesInt32(i.Dims),
	}
	if i.Values == nil {
		return vol, nil
	}
	goptr, err := gocudnn.MakeGoPointer(i.Values)
	if err != nil {
		vol.Destroy()
		return nil, err
	}
	err = vol.LoadMem(goptr)
	if err != nil {
		vol.Destroy()
		return nil, err
	}
	return vol, nil
}
