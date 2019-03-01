package batchnorm

import (
	"fmt"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
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
	RRM               []byte                `json:"RRM"`
	RRV               []byte                `json:"RRV"`
	RSM               []byte                `json:"RSM"`
	RSV               []byte                `json:"RSV"`
}

//Info returns the Info struct
func (o *Ops) Info() (Info, error) {
	sizet, err := o.bnsbmvd.GetSizeInBytes()
	if err != nil {
		return Info{}, err
	}

	rrm := make([]byte, sizet)
	rrv := make([]byte, sizet)
	rsm := make([]byte, sizet)
	rsv := make([]byte, sizet)
	rmmwritten, err := o.rrm.Write(rrm)
	if err != nil {
		fmt.Println("Bytes Written rrm: ", rmmwritten)
		return Info{}, err
	}
	rmmwritten, err = o.rrv.Write(rrv)
	if err != nil {
		fmt.Println("Bytes Written rrv: ", rmmwritten)
		return Info{}, err
	}
	rmmwritten, err = o.rsm.Write(rsm)
	if err != nil {
		fmt.Println("Bytes Written rsm: ", rmmwritten)
		return Info{}, err
	}
	rmmwritten, err = o.rsv.Write(rsv)
	if err != nil {
		fmt.Println("Bytes Written rsv: ", rmmwritten)
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
