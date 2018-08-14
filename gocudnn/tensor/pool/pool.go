package pool

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Ops holds what the data that is need to perform the pooling operations
type Ops struct {
	desc     *gocudnn.PoolingD
	hlpr     gocudnn.Pooling
	inputdim int
}

//Info is contains all the necessarry information to build a pooling Ops.
type Info struct {
	Mode      gocudnn.PoolingMode    `json:"Mode"`
	Nan       gocudnn.PropagationNAN `json:"NAN"`
	InputDims int                    `json:"InputDims"`
	Window    []int32                `json:"Window"`
	Padding   []int32                `json:"Padding"`
	Stride    []int32                `json:"Stride"`
}

//Build builds and returns an *Op from the info already stored in the Info type.
func (input Info) Build() (*Ops, error) {
	if input.InputDims > 4 {
		pooldim := int32(len(input.Window))
		desc, err := gocudnn.Pooling{}.CreatePoolingNdDescriptor(input.Mode, input.Nan, pooldim, input.Window, input.Padding, input.Stride)
		if err != nil {
			return nil, err
		}
		return &Ops{
			desc:     desc,
			inputdim: input.InputDims,
		}, nil

	}
	if input.InputDims < 4 {
		return nil, errors.New("Dims should be 4 or more")
	}
	desc, err := gocudnn.Pooling{}.NewPooling2dDescriptor(input.Mode, input.Nan, input.Window, input.Padding, input.Stride)
	if err != nil {
		return nil, err
	}
	return &Ops{
		desc:     desc,
		inputdim: input.InputDims,
	}, nil
}

//Build builds the pooling ops
func Build(mode gocudnn.PoolingMode, nan gocudnn.PropagationNAN, input *tensor.Volume, window, padding, stride []int32) (*Ops, error) {
	_, _, dims, err := input.Properties()
	if err != nil {
		return nil, err
	}
	if len(dims) > 4 {
		pooldim := int32(len(window)) //This is probably wrong. I haven't tested it yet
		desc, err := gocudnn.Pooling{}.CreatePoolingNdDescriptor(mode, nan, pooldim, window, padding, stride)
		if err != nil {
			return nil, err
		}
		return &Ops{
			desc:     desc,
			inputdim: len(dims),
		}, nil
	}
	if len(dims) < 4 {
		return nil, errors.New("Dims should be 4 or more")
	}
	desc, err := gocudnn.Pooling{}.NewPooling2dDescriptor(mode, nan, window, padding, stride)
	if err != nil {
		return nil, err
	}
	return &Ops{
		desc:     desc,
		inputdim: len(dims),
	}, nil
}

//Properties returns the pooling properties
func (p *Ops) Properties() (gocudnn.PoolingMode, gocudnn.PropagationNAN, []int32, []int32, []int32, error) {
	return p.desc.GetPoolingDescriptor()

}

//Info returns the info struct usually used for saving the info to a jason format
func (p *Ops) Info() (Info, error) {
	mode, nan, window, pad, stride, err := p.desc.GetPoolingDescriptor()
	inputdim := p.inputdim
	return Info{
		Mode:      mode,
		Nan:       nan,
		Window:    window,
		Padding:   pad,
		Stride:    stride,
		InputDims: inputdim,
	}, err

}

//OutputDims returns the dims the output wil have considering the input
func (p *Ops) OutputDims(input *tensor.Volume) ([]int32, error) {
	return p.desc.GetPoolingForwardOutputDim(input.TD())
}

//FwdProp does the pooling fwd operation
func (p *Ops) FwdProp(handle *gocudnn.Handle, alpha, beta float64, x, y *tensor.Volume) error {
	_, dtype, _, err := x.Properties()
	if err != nil {
		return err
	}
	var a gocudnn.CScalar
	var b gocudnn.CScalar
	t := tensor.Flags()
	switch dtype {
	case t.Data.Double():
		a = gocudnn.CDouble(alpha)
		b = gocudnn.CDouble(beta)
	case t.Data.Float():
		a = gocudnn.CFloat(alpha)
		b = gocudnn.CFloat(beta)
	case t.Data.Int32():
		a = gocudnn.CInt(alpha)
		b = gocudnn.CInt(beta)
	case t.Data.Int8():
		a = gocudnn.CInt8(alpha)
		b = gocudnn.CInt8(beta)
	case t.Data.UInt8():
		a = gocudnn.CUInt8(alpha)
		b = gocudnn.CUInt8(beta)

	default:
		return errors.New("Not supported Format")
	}

	return p.hlpr.Funcs.PoolingForward(handle, p.desc, a, x.TD(), x.Memer(), b, y.TD(), y.Memer())
}

//BwdProp does the backward propagation operation
func (p *Ops) BwdProp(handle *gocudnn.Handle, alpha, beta float64, x, dx, y, dy *tensor.Volume) error {
	_, dtype, _, err := x.Properties()
	if err != nil {
		return err
	}

	var a gocudnn.CScalar
	var b gocudnn.CScalar
	t := tensor.Flags()
	switch dtype {
	case t.Data.Double():
		a = gocudnn.CDouble(alpha)
		b = gocudnn.CDouble(beta)
	case t.Data.Float():
		a = gocudnn.CFloat(alpha)
		b = gocudnn.CFloat(beta)
	case t.Data.Int32():
		a = gocudnn.CInt(alpha)
		b = gocudnn.CInt(beta)
	case t.Data.Int8():
		a = gocudnn.CInt8(alpha)
		b = gocudnn.CInt8(beta)
	case t.Data.UInt8():
		a = gocudnn.CUInt8(alpha)
		b = gocudnn.CUInt8(beta)

	default:
		return errors.New("Not supported Format")
	}

	return p.hlpr.Funcs.PoolingBackward(handle, p.desc, a, y.TD(), y.Memer(), dy.TD(), dy.Memer(), x.TD(), x.Memer(), b, dx.TD(), dx.Memer())

}

//Destroy will destroy the Pooling Descriptor
func (p *Ops) Destroy() error {
	return p.Destroy()
}
