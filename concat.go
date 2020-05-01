package gocunets

import (
	"errors"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/tensor"
)

//ReverseConcat is just a simple solution to split a source into multiple dests. If the source channel is not divisible by
//the number of dests then the remainder of the sources channels will be set into the last dest.
type ReverseConcat struct {
	c          *tensor.Concat
	h          *Handle
	dests      []*tensor.Volume
	deltadests []*tensor.Volume
	src        *tensor.Volume
	deltasrc   *tensor.Volume
}

//CreateReverseConcat creates a reverse concat
func CreateReverseConcat(h *Handle) (c *ReverseConcat, err error) {
	c = new(ReverseConcat)
	c.c, err = tensor.CreateConcat(h.Handler)
	c.h = h
	return c, err
}

//SetOutputDeltaDests sets the delta dests for back propagation
func (c *ReverseConcat) SetOutputDeltaDests(deltadests []*Tensor) {
	c.deltadests = make([]*tensor.Volume, len(deltadests))
	for i := range deltadests {
		c.deltadests[i] = deltadests[i].Volume
	}
	return
}

//SetOutputDests sets the output dests
func (c *ReverseConcat) SetOutputDests(dests []*Tensor) {
	c.dests = make([]*tensor.Volume, len(dests))
	for i := range dests {
		c.dests[i] = dests[i].Volume
	}
	return
}

//SetInputSource sets the input source
func (c *ReverseConcat) SetInputSource(src *Tensor) {
	c.src = src.Volume
}

//SetInputDeltaSource sets the delta src for back propagation
func (c *ReverseConcat) SetInputDeltaSource(deltasrc *Tensor) {
	c.deltasrc = deltasrc.Volume
}

//FindOutputDimsfromInputDims finds the input dims from output dims. Last dest will get + the remainder for overflow or just the remainder of underflow
func (c *ReverseConcat) FindOutputDimsfromInputDims(src []int32, ndests int32, frmt TensorFormat) (destdims [][]int32, err error) {
	fflg := frmt

	switch frmt {
	case fflg.NCHW():
		channels := src[1]
		destchansize := channels / ndests
		remainder := channels % ndests
		var underflow bool
		if destchansize*ndests > channels {
			underflow = true
		}
		destdims := make([][]int32, ndests)
		for i := int32(0); i < ndests; i++ {

			destdims[i] = make([]int32, len(src))
			for j := range destdims[i] {
				destdims[i][j] = src[j]
			}

			if i == int32(len(destdims)-1) {
				if underflow {
					destdims[i][1] = remainder
				}
				destdims[i][1] = destchansize + remainder
			} else {
				destdims[i][1] = destchansize
			}
		}
		return destdims, nil
	case fflg.NHWC():
		channels := src[len(src)-1]
		destchansize := channels / ndests
		remainder := channels % ndests
		var underflow bool
		if destchansize*ndests > channels {
			underflow = true
		}
		for i := int32(0); i < ndests; i++ {

			destdims[i] = make([]int32, len(src))
			for j := range destdims[i] {
				destdims[i][j] = src[j]
			}

			if i == int32(len(destdims)-1) {
				if underflow {
					destdims[i][len(src)-1] = remainder
				}
				destdims[i][len(src)-1] = destchansize + remainder
			} else {
				destdims[i][len(src)-1] = destchansize
			}
		}
		return destdims, nil
	default:
		return nil, errors.New("(c *ReverseConcat) FindOutputDimsfromInputDims: unsupported format")
	}

}

//FindOutputDims finds the output dims for the dests
func (c *ReverseConcat) FindOutputDims(Source *Tensor, ndests int32) (outputdims [][]int32, err error) {

	var tf TensorFormat
	tf.TensorFormat = Source.Format()
	return c.FindOutputDimsfromInputDims(Source.Dims(), ndests, tf)

}

//Forward Does forward with data flowing srcs to dest
func (c *ReverseConcat) Forward() error {

	return c.c.Backward(c.h.Handler, c.dests, c.src)
}

//Backward Does backward with data flowing dest to srcs
func (c *ReverseConcat) Backward() error {
	return c.c.Forward(c.h.Handler, c.deltadests, c.deltasrc)
}

//Concat does the concat operation
type Concat struct {
	c         *tensor.Concat
	h         *Handle
	srcs      []*tensor.Volume
	deltasrcs []*tensor.Volume
	dest      *tensor.Volume
	deltadest *tensor.Volume
}

//CreateConcat creates a concat operation handler
func CreateConcat(h *Handle) (c *Concat, err error) {
	c = new(Concat)
	c.c, err = tensor.CreateConcat(h.Handler)
	c.h = h
	return c, err

}

//SetInputDeltaSrcs sets the delta srcs for back propagation
func (c *Concat) SetInputDeltaSrcs(deltasrcs []*Tensor) {
	c.deltasrcs = make([]*tensor.Volume, len(deltasrcs))
	for i := range deltasrcs {
		c.deltasrcs[i] = deltasrcs[i].Volume
	}
	return
}

//SetInputSrcs sets the input srcs
func (c *Concat) SetInputSrcs(srcs []*Tensor) {
	c.srcs = make([]*tensor.Volume, len(srcs))
	for i := range srcs {
		c.srcs[i] = srcs[i].Volume
	}
	return
}

//SetDest sets the output dest
func (c *Concat) SetDest(dest *Tensor) {
	c.dest = dest.Volume
}

//SetDeltaDest sets the delta dest for back propagation
func (c *Concat) SetDeltaDest(deltadest *Tensor) {
	c.deltadest = deltadest.Volume
}

//FindOutputDimsfromInputDims finds the input dims from output dims
func (c *Concat) FindOutputDimsfromInputDims(srcs [][]int32, frmt TensorFormat) (outputdims []int32, err error) {

	return c.c.GetOutputDimsfromInputDims(srcs, frmt.TensorFormat)
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

//Forward Does forward with data flowing srcs to dest
func (c *Concat) Forward() error {

	return c.c.Forward(c.h.Handler, c.srcs, c.dest)
}

//Backward Does backward with data flowing dest to srcs
func (c *Concat) Backward() error {
	return c.c.Backward(c.h.Handler, c.deltasrcs, c.deltadest)
}
