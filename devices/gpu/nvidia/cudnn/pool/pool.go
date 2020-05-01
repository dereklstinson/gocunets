package pool

import (
	"errors"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/gocudnn"
)

//Ops holds what the data that is need to perform the pooling operations
type Ops struct {
	desc    *gocudnn.PoolingD
	reverse bool
	//inputdim int
}

//OpInfo is contains all the necessarry information to build a pooling Ops.
type OpInfo struct {
	Mode    gocudnn.PoolingMode `json:"Mode"`
	Nan     gocudnn.NANProp     `json:"NAN"`
	Window  []int32             `json:"Window"`
	Padding []int32             `json:"Padding"`
	Stride  []int32             `json:"Stride"`
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

//StageOperationReverse builds a reverse pooling operation.  Experemental!!!!
//
//Will be testing soon.  This is just running the backward pooling when forward is called.
//and it is runing forward pooling when backward is called.
func StageOperationReverse(mode gocudnn.PoolingMode, nan gocudnn.NANProp, window, padding, stride []int32) (*Ops, error) {
	desc, err := gocudnn.CreatePoolingDescriptor()
	if err != nil {
		return nil, err
	}
	err = desc.Set(mode, nan, window, padding, stride)
	if err != nil {
		return nil, err
	}
	return &Ops{
		desc:    desc,
		reverse: true,
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
	if p.reverse {

		_, _, w, p, s, err := p.desc.Get()
		if err != nil {
			return nil, err
		}
		x := make([]int32, len(input.Dims()))
		copy(x, input.Dims())
		frmt := input.Format()
		flg := frmt
		switch frmt {
		case flg.NCHW():
			newdims := getmdoutputdim(x[2:], p, w, s)
			if len(newdims) != len(x[2:]) {
				panic("error in backward output algo")
			}
			for i := 2; i < len(x); i++ {
				x[i] = newdims[i]
			}
			return x, nil
		case flg.NHWC():
			newdims := getmdoutputdim(x[1:len(x)-2], p, w, s)
			if len(newdims) != len(x[1:len(x)-2]) {
				panic("error in backward output algo")
			}
			for i := 1; i < len(x)-1; i++ {
				x[i] = newdims[i]
			}
			return x, nil
		default:
			return nil, errors.New("Unsupported Format")
		}

	}
	return p.desc.GetOutputDims(input.TD())
}
func getmdoutputdim(x, p, w, s []int32) []int32 {
	opdims := make([]int32, len(x))
	for i := range opdims {
		opdims[i] = onedoutputdim(x[i], p[i], w[i], s[i])
	}
	return opdims
}
func onedoutputdim(x, p, w, s int32) int32 {
	return (x-1)*s - 2*p + w
}

//Forward does the pooling fwd operation
func (p *Ops) Forward(handle *cudnn.Handler, alpha, beta float64, x, y *tensor.Volume) error {
	if p.reverse {
		return p.desc.Backward(handle.Cudnn(), alpha, x.TD(), x, x.TD(), x, y.TD(), y, beta, y.TD(), y)
	}
	return p.desc.Forward(handle.Cudnn(), alpha, x.TD(), x, beta, y.TD(), y)
}

//Backward does the backward propagation operation
func (p *Ops) Backward(handle *cudnn.Handler, alpha, beta float64, x, dx, y, dy *tensor.Volume) error {
	if p.reverse {
		return p.desc.Forward(handle.Cudnn(), alpha, dy.TD(), dy, beta, dx.TD(), dx)
	}
	return p.desc.Backward(handle.Cudnn(), alpha, y.TD(), y, dy.TD(), dy, x.TD(), x, beta, dx.TD(), dx)

}

//Destroy will destroy the Pooling Descriptor
func (p *Ops) Destroy() error {
	return p.Destroy()
}
