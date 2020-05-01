package tensor

import (
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocudnn/xtra"
	"github.com/dereklstinson/cutil"

	gocudnn "github.com/dereklstinson/gocudnn"
)

//Concat concats the channels of multiple tensors into a new tensor. Concats are seperated by batch.
type Concat struct {
	//	h  *cudnn.Handler
	c  *xtra.ConcatEx
	fa float64
	fb float64
	ba float64
	bb float64
}

//CreateConcat creates a concat handler.  It contains the kernel that does the concat operation on the gpu
//Default alpha, beta are set to alpha = 1, beta =0.
func CreateConcat(h *cudnn.Handler) (c *Concat, err error) {
	c = new(Concat)

	c.c, err = xtra.CreateConcatEx(h.XHandle())
	c.fa, c.ba, c.fb, c.bb = 1, 1, 0, 0
	return c, err
}

//SetForwardAlpha sets the forward alpha
func (c *Concat) SetForwardAlpha(alpha float64) {
	c.fa = alpha
}

//SetForwardBeta sets the forward beta
func (c *Concat) SetForwardBeta(beta float64) {
	c.fb = beta
}

//SetBackwardAlpha sets the backward alpha
func (c *Concat) SetBackwardAlpha(alpha float64) {
	c.ba = alpha
}

//SetBackwardBeta sets the backward beta
func (c *Concat) SetBackwardBeta(beta float64) {
	c.bb = beta
}

//GetOutputDimsfromInputDims gets the outputdims from the inputdims.  If srcs dims other than the channel dims not equal. Function will return an error.
func (c *Concat) GetOutputDimsfromInputDims(srcs [][]int32, frmt gocudnn.TensorFormat) ([]int32, error) {
	return c.c.GetOutputDimsFromInputDims(srcs, frmt)
}

//GetOutputdims returns the output dims of the tensor for the outputed Volume.
func (c *Concat) GetOutputdims(srcs []*Volume) (outputdims []int32, err error) {
	descriptors := make([]*gocudnn.TensorD, len(srcs))
	for i := range srcs {
		descriptors[i] = srcs[i].TD()
	}
	outputdims, err = c.c.GetOutputdims(descriptors)
	return outputdims, err
}

//Forward does the forward where data in srcs goes to dest
func (c *Concat) Forward(h *cudnn.Handler, srcs []*Volume, dest *Volume) error {
	sdescriptors := make([]*gocudnn.TensorD, len(srcs))
	smemory := make([]cutil.Mem, len(srcs))
	for i := range srcs {
		sdescriptors[i] = srcs[i].TD()
		smemory[i] = srcs[i].Malloced
	}
	return c.c.Op(h.XHandle(), sdescriptors, smemory, c.fa, dest.TD(), dest, c.fb, true)
}

//Backward does the backward algorithm where the data in dest goes to the srcs.
func (c *Concat) Backward(h *cudnn.Handler, srcs []*Volume, dest *Volume) error {
	sdescriptors := make([]*gocudnn.TensorD, len(srcs))
	smemory := make([]cutil.Mem, len(srcs))
	for i := range srcs {
		sdescriptors[i] = srcs[i].TD()
		smemory[i] = srcs[i].Malloced
	}
	return c.c.Op(h.XHandle(), sdescriptors, smemory, c.ba, dest.TD(), dest, c.bb, false)
}
