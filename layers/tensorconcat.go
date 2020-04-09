package layers

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
)

//Concat does the concat operation
type Concat struct {
	c *tensor.Concat
}

//CreateConcat creates a concat operation handler
func CreateConcat(h *cudnn.Handler) (c *Concat, err error) {
	c = new(Concat)
	c.c, err = tensor.CreateConcat(h)
	return c, err

}

//FindOutputDims finds the output dims
func (c *Concat) FindOutputDims(srcs []*Tensor) (outputdims []int32, err error) {
	vols := make([]*tensor.Volume, len(srcs))
	for i := range vols {
		vols[i] = srcs[i].Volume
	}
	outputdims, err = c.c.GetOutputdims(vols)
	return outputdims, err
}

//Forward does the forward concat srcs to dest
func (c *Concat) Forward(h *cudnn.Handler, srcs []*Tensor, dest *Tensor) error {
	svol := make([]*tensor.Volume, len(srcs))
	for i := range srcs {
		svol[i] = srcs[i].Volume
	}
	return c.c.Forward(h, svol, dest.Volume)
}

//Backward does the backward concat dest to srcs
func (c *Concat) Backward(h *cudnn.Handler, srcs []*Tensor, dest *Tensor) error {
	svol := make([]*tensor.Volume, len(srcs))
	for i := range srcs {
		svol[i] = srcs[i].Volume
	}
	return c.c.Backward(h, svol, dest.Volume)
}
