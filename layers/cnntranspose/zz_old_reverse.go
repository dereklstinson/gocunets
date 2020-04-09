package cnntranspose

/*
OLD

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

func (l *Layer) SetForwardScalars(alpha, beta float64) {
	l.conv.SetBackwardScalars(alpha, beta)
}

//SetBackwardScalars sets the backward scalars for the back propigation
func (l *Layer) SetBackwardScalars(alpha, beta float64) {
	l.conv.SetForwardScalars(alpha, beta)
}

//SetOtherScalars sets the parameterscalars of the convolution operation
func (l *Layer) SetOtherScalars(alpha, beta float64) {
	l.conv.SetOtherScalars(alpha, beta)
}


//ReverseBuild sets up reverse version of cnn transpose
//The output is determined by  -->  output = (slide *(input-1)) - (2*padding) + (((filter-1)*dilation)+1)
func ReverseBuild(handle *cudnn.Handler,
	frmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
	mtype gocudnn.MathType,
	groupcount int32,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad, //largestgains with no pad
	stride, //largestgains with more stride
	dilation []int32, //largest gains with dilation
	seed uint64) (*Layer, error) {
	conv, err := cnn.SetupReverse(handle, frmt, dtype, mtype, groupcount, filterdims, convmode, pad, stride, dilation, seed)
	if err != nil {
		return nil, err
	}

	if err != nil {
		return nil, err
	}
	return &Layer{
		conv: conv,
		mode: convtransposereverse,
	}, nil
}

func (l *Layer) reverseForwardProp(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.Tensor) error {
	return l.conv.ReverseForwardProp(handle, wspace, x, y)
}
func (l *Layer) reverseBackPropFilterData(handle *cudnn.Handler, wspacedata, wspacefilter *nvidia.Malloced, x, dx, dy *layers.Tensor) error {
	return l.conv.ReverseBackPropFilterData(handle, wspacedata, wspacefilter, x, dx, dy)
}
func (l *Layer) reverseBackPropData(handle *cudnn.Handler, wspace *nvidia.Malloced, dx, dy *layers.Tensor) error {
	return l.conv.ReverseBackPropData(handle, wspace, dx, dy)
}
func (l *Layer) reverseBackPropFilter(handle *cudnn.Handler, wspace *nvidia.Malloced, x, dy *layers.Tensor) error {
	return l.conv.ReverseBackPropFilter(handle, wspace, x, dy)
}

func (l *Layer) reverseOutput(handle *cudnn.Handler, input *layers.Tensor) (*layers.Tensor, error) {
	return l.conv.MakeReverseOutputTensor(handle, input)
}

*/
