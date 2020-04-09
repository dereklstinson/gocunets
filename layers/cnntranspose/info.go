package cnntranspose

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/deconvolution"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
)

//"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/convolution"
//	"github.com/dereklstinson/GoCuNets/layers"

//Info contains the info that is needed to build a CNN layer
type Info struct {
	DeConvolution deconvolution.Info `json:"convolution,omitempty"`
	Weights       tensor.Info        `json:"weights,omitempty"`
	Bias          tensor.Info        `json:"bias,omitempty"`
}

//Info returns the info struct for the convolution
func (l *Layer) Info() (Info, error) {
	cinfo, err := l.conv.Info()
	if err != nil {
		return Info{}, err
	}
	winfo, err := l.w.Info()
	if err != nil {
		return Info{}, err
	}
	binfo, err := l.bias.Info()
	if err != nil {
		return Info{}, err
	}
	return Info{
		DeConvolution: cinfo,
		Weights:       winfo,
		Bias:          binfo,
	}, nil
}
