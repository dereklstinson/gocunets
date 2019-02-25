package cnntranspose

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/Nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Info contains information to build a layer
type Info struct {
	ReverseInfo cnn.Info `json:"reverse_info,omitempty"`
}

//SetAllScalars sets all the scalars.  This is going to be used for a PSO.
//Order to put scalars each operation has 3 scalars in each there will be alaph, alpha2, beta
//the order the array will need to be is fwd, bwd-data,and bwd-filter
func (l *Layer) SetAllScalars(fwd3bwdd3bwdf3 []float64) error {
	return l.conv.SetAllScalars(fwd3bwdd3bwdf3)
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
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
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

func (l *Layer) reverseForwardProp(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	return l.conv.ReverseForwardProp(handle, wspace, x, y)
}
func (l *Layer) reverseBackPropFilterData(handle *cudnn.Handler, wspacedata, wspacefilter *gocudnn.Malloced, x, y *layers.IO) error {
	return l.conv.ReverseBackPropFilterData(handle, wspacedata, wspacefilter, x, y)
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
