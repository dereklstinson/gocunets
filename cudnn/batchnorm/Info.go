package batchnorm

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Info is used to save the layer for later use.
type Info struct {
	Epsilon           float64               `json:"Epsilon"`
	Exponentialfactor uint                  `json:"Exponentialfactor"`
	Mode              gocudnn.BatchNormMode `json:"Mode"`
	Format            cudnn.TensorFormat    `json:"Format"`
	DataType          cudnn.DataType        `json:"DataType"`
	Nan               cudnn.NanMode         `json:"Nan"`
	Dims              []int32               `json:"Dims"`
	Stride            []int32               `json:"Stride"`
	RRM               []float64             `json:"RRM"`
	RRV               []float64             `json:"RRV"`
	RSM               []float64             `json:"RSM"`
	RSV               []float64             `json:"RSV"`
}

//Info returns the Info struct
func (o *Ops) Info() (Info, error) {

	rrm, err := getfloat64val(o.bnsbmvd, o.rrm)
	if err != nil {
		return Info{}, err
	}
	rrv, err := getfloat64val(o.bnsbmvd, o.rrv)
	if err != nil {
		return Info{}, err
	}
	rsm, err := getfloat64val(o.bnsbmvd, o.rsm)
	if err != nil {
		return Info{}, err
	}
	rsv, err := getfloat64val(o.bnsbmvd, o.rsv)
	if err != nil {
		return Info{}, err
	}
	dtype, dims, stride, err := o.bnsbmvd.GetDescrptor()

	if err != nil {
		return Info{}, err
	}
	frmt, err := o.bnsbmvd.GetFormat()
	if err != nil {
		return Info{}, err
	}
	return Info{
		Epsilon:  o.epsilon,
		Mode:     o.mode,
		DataType: cudnn.DataType(dtype),
		Stride:   stride,
		Dims:     dims,
		Format:   cudnn.TensorFormat(frmt),
		RRM:      rrm,
		RRV:      rrv,
		RSV:      rsv,
		RSM:      rsm,
	}, nil

}

func getfloat64val(desc *gocudnn.TensorD, mem *gocudnn.Malloced) ([]float64, error) {
	var dflgs gocudnn.DataTypeFlag
	dtype, dims, stride, err := desc.GetDescrptor()
	if err != nil {
		return nil, err
	}
	size := utils.FindVolumeInt32(dims, stride)
	vals := make([]float64, size)
	switch dtype {
	case dflgs.Double():

		values := make([]float64, size)
		err = mem.FillSlice(values)
		if err != nil {
			return nil, err
		}
		for i := range values {
			vals[i] = float64(values[i])
		}

	case dflgs.Float():
		values := make([]float32, size)
		err = mem.FillSlice(values)
		if err != nil {
			return nil, err
		}
		for i := range values {
			vals[i] = float64(values[i])
		}
	case dflgs.Int32():
		values := make([]int32, size)
		err = mem.FillSlice(values)
		if err != nil {
			return nil, err
		}
		for i := range values {
			vals[i] = float64(values[i])
		}
	case dflgs.Int8():
		values := make([]float64, size)
		err = mem.FillSlice(values)
		if err != nil {
			return nil, err
		}
		for i := range values {
			vals[i] = float64(values[i])
		}

	default:
		return nil, errors.New("Unsupported Format : Most likely internal error. Contact Code Writer")
	}
	return vals, nil

}
