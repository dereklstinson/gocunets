package cnntranspose

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Info contains information to build a layer
type Info struct {
	ReverseInfo cnn.Info `json:"reverse_info,omitempty"`
}

//SetAlphaScalars sets the alpha scalars in the order of fwd, bwd-data, and bwd-filt.
func (l *Layer) SetAlphaScalars(alphas []float64) error {
	return l.conv.SetAlphaScalars(alphas)
}

//SetBetaScalars sets the beta scalars in the order of fwd, bwd-data, and bwd-filt.
func (l *Layer) SetBetaScalars(alphas []float64) error {
	return l.conv.SetBetaScalars(alphas)
}

//NumAlphaScalars returns the number of alpha scalars for the forward,backward data, and backward filter
func (l *Layer) NumAlphaScalars() int {
	return l.conv.NumAlphaScalars()
}

//NumBetaScalars returns the number of beta scalars for the forward,backward data, and backward filter
func (l *Layer) NumBetaScalars() int {
	return l.conv.NumBetaScalars()
}

//Info method returns an info struct used to save and stuff
func (l *Layer) Info() (Info, error) {
	info, err := l.conv.Info()
	if err != nil {
		return Info{}, err
	}
	return Info{
		ReverseInfo: info,
	}, nil
}

//ReverseBuild sets up reverse version of cnn transpose
//The output is determined by  -->  output = (slide *(input-1)) - (2*padding) + (((filter-1)*dilation)+1)
func ReverseBuild(handle *cudnn.Handler,
	frmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad, //largestgains with no pad
	stride, //largestgains with more stride
	dilation []int32, //largest gains with dilation
	inputlayer bool,
	seed uint64) (*Layer, error) {
	conv, err := cnn.SetupReverse(handle, frmt, dtype, filterdims, convmode, pad, stride, dilation, seed)
	if err != nil {
		return nil, err
	}

	if err != nil {
		return nil, err
	}
	return &Layer{
		conv:       conv,
		mode:       convtransposereverse,
		inputlayer: inputlayer,
	}, nil
}

func (l *Layer) reverseForwardProp(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.IO) error {
	return l.conv.ReverseForwardProp(handle, wspace, x, y)
}
func (l *Layer) reverseBackPropFilterData(handle *cudnn.Handler, wspacedata, wspacefilter *nvidia.Malloced, x, y *layers.IO) error {
	return l.conv.ReverseBackPropFilterData(handle, wspacedata, wspacefilter, x, y)
}
func (l *Layer) reverseBackPropData(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.IO) error {
	return l.conv.ReverseBackPropData(handle, wspace, x, y)
}
func (l *Layer) reverseBackPropFilter(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.IO) error {
	return l.conv.ReverseBackPropFilter(handle, wspace, x, y)
}

func (l *Layer) reverseOutput(handle *cudnn.Handler, input *layers.IO) (*layers.IO, error) {
	return l.conv.MakeReverseOutputTensor(handle, input)
}
func (l *Layer) reverseOutputInference(handle *cudnn.Handler, input *layers.IO) (*layers.IO, error) {
	return l.conv.MakeReverseOutputTensorInference(handle, input)
}
