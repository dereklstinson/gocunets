package batchnorm

import (
	"io/ioutil"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Info is used to save the layer for later use.
type Info struct {
	Epsilon           float64               `json:"Epsilon"`
	Exponentialfactor uint                  `json:"Exponentialfactor"`
	Mode              gocudnn.BatchNormMode `json:"Mode"`
	Format            gocudnn.TensorFormat  `json:"Format"`
	DataType          gocudnn.DataType      `json:"DataType"`
	Nan               gocudnn.NANProp       `json:"Nan"`
	Dims              []int32               `json:"Dims"`
	Stride            []int32               `json:"Stride"`
	RRM               []byte                `json:"RRM"`
	RRV               []byte                `json:"RRV"`
	RSM               []byte                `json:"RSM"`
	RSV               []byte                `json:"RSV"`
}

//Info returns the Info struct
func (o *Ops) Info(h *cudnn.Handler) (Info, error) {
	s := h.Stream()
	rrmw := o.rrm.NewReadWriter(s)

	rrvw := o.rrv.NewReadWriter(s)

	rsmw := o.rsm.NewReadWriter(s)

	rsvw := o.rsv.NewReadWriter(s)

	rrm, err := ioutil.ReadAll(rrmw)
	if err != nil {
		return Info{}, err
	}
	rrv, err := ioutil.ReadAll(rrvw)
	if err != nil {
		return Info{}, err
	}
	rsm, err := ioutil.ReadAll(rsmw)
	if err != nil {
		return Info{}, err
	}
	rsv, err := ioutil.ReadAll(rsvw)
	if err != nil {
		return Info{}, err
	}
	frmt, dtype, dims, stride, err := o.bnsbmvd.Get()

	if err != nil {
		return Info{}, err
	}
	mode, err := o.op.Get()
	if err != nil {
		return Info{}, err
	}
	return Info{
		Epsilon:  o.epsilon,
		Mode:     mode,
		DataType: dtype,
		Stride:   stride,
		Dims:     dims,
		Format:   frmt,
		RRM:      rrm,
		RRV:      rrv,
		RSV:      rsv,
		RSM:      rsm,
	}, nil

}
