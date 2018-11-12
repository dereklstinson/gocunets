package cnntranspose

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//ReverseBuild sets up reverse version of cnn transpose
func ReverseBuild(handle *cudnn.Handler,
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
	inputdimsguess []int32, //UpscaledDims will be the dims of the input before the convolution
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad, //largestgains with no pad
	stride, //largestgains with more stride
	dilation []int32, //largest gains with dilation
	inputlayer bool,
	managedmem bool) (*Layer, error) {
	conv, err := cnn.SetupDynamicReverse(handle, frmt, dtype, inputdimsguess, filterdims, convmode, pad, stride, dilation, managedmem)
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

func (l *Layer) reverseForwardProp(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	return l.conv.ReverseForwardProp(handle, wspace, x, y)
}
func (l *Layer) reverseBackPropFilterData(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	return l.conv.ReverseBackPropFilterData(handle, wspace, x, y)
}
func (l *Layer) reverseBackPropData(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	return l.conv.ReverseBackPropData(handle, wspace, x, y)
}
func (l *Layer) reverseBackPropFilter(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	return l.conv.ReverseBackPropFilter(handle, wspace, x, y)
}

func (l *Layer) reverseOutput(handle *cudnn.Handler, input *layers.IO) (*layers.IO, error) {
	return l.conv.MakeReverseOutputTensor(handle, input)
}
