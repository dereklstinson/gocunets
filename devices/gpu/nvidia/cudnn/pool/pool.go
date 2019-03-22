package pool

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Ops holds what the data that is need to perform the pooling operations
type Ops struct {
	desc *gocudnn.PoolingD
	//inputdim int
}

//OpInfo is contains all the necessarry information to build a pooling Ops.
type OpInfo struct {
	Mode gocudnn.PoolingMode `json:"Mode"`
	Nan  gocudnn.NANProp     `json:"NAN"`
	//InputDims int                 `json:"InputDims"`
	Window  []int32 `json:"Window"`
	Padding []int32 `json:"Padding"`
	Stride  []int32 `json:"Stride"`
}

//Stage builds and returns an *Op from the info already stored in the Info type.
func (input OpInfo) Stage() (*Ops, error) {
	desc, err := gocudnn.CreatePoolingDescriptor()
	if err != nil {
		return nil, err
	}
	err = desc.Set(input.Mode, input.Nan, input.Window, input.Padding, input.Stride)
	if err != nil {
		return nil, err
	}

	return &Ops{
		desc: desc,
		//	inputdim: input.InputDims,
	}, nil
}

//StageOperation builds the pooling ops
func StageOperation(mode gocudnn.PoolingMode, nan gocudnn.NANProp, window, padding, stride []int32) (*Ops, error) {
	desc, err := gocudnn.CreatePoolingDescriptor()
	if err != nil {
		return nil, err
	}
	err = desc.Set(mode, nan, window, padding, stride)
	if err != nil {
		return nil, err
	}

	return &Ops{
		desc: desc,
		//	inputdim: input.InputDims,
	}, nil
}

//Properties returns the pooling properties
func (p *Ops) Properties() (gocudnn.PoolingMode, gocudnn.NANProp, []int32, []int32, []int32, error) {
	return p.desc.Get()

}

//Info returns the info struct usually used for saving the info to a jason format
func (p *Ops) Info() (OpInfo, error) {
	mode, nan, window, pad, stride, err := p.desc.Get()

	return OpInfo{
		Mode:    mode,
		Nan:     nan,
		Window:  window,
		Padding: pad,
		Stride:  stride,
	}, err

}

//OutputDims returns the dims the output wil have considering the input
func (p *Ops) OutputDims(input *tensor.Volume) ([]int32, error) {
	return p.desc.GetOutputDims(input.TD())
}

//Forward does the pooling fwd operation
func (p *Ops) Forward(handle *cudnn.Handler, alpha, beta float64, x, y *tensor.Volume) error {

	return p.desc.Forward(handle.Cudnn(), alpha, x.TD(), x.Memer(), beta, y.TD(), y.Memer())
}

//BackWard does the backward propagation operation
func (p *Ops) BackWard(handle *cudnn.Handler, alpha, beta float64, x, dx, y, dy *tensor.Volume) error {

	return p.desc.Backward(handle.Cudnn(), alpha, y.TD(), y.Memer(), dy.TD(), dy.Memer(), x.TD(), x.Memer(), beta, dx.TD(), dx.Memer())

}

//Destroy will destroy the Pooling Descriptor
func (p *Ops) Destroy() error {
	return p.Destroy()
}
