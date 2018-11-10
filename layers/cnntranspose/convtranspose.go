package cnntranspose

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/reshapes"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

/*
There is a few ways to do this.
1) using transformtensor and increase the size of the tensor. If done correctly you can make it so that every other value has 0.  //This will have to hold the output in mem
2) Using batch to shape then shape to batch.
3) Use resize and then do a resize back prop which back propigates the errors to the source pixel.
*/

//Layer contains the ops need for ConvTranspose
type Layer struct {
	conv         *cnn.Layer
	trans        *reshapes.Ops
	hiddenmem    *layers.IO
	mode         convtransposemode
	originaldims []int32
	outputdims   []int32
}
type convtransposemode int

const (
	convtransposetrans = convtransposemode(1)
)

//Transform sets up a transform version of cnn transpose
func Transform(handle *cudnn.Handler,
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
	upscaleddims []int32, //UpscaledDims will be the dims of the input before the convolution
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	managedmem bool) (*Layer, error) {
	conv, err := cnn.SetupDynamic(handle, frmt, dtype, filterdims, filterdims, convmode, pad, stride, dilation, managedmem)
	if err != nil {
		return nil, err
	}
	vol, err := layers.Build(frmt, dtype, upscaleddims, managedmem)
	if err != nil {
		return nil, err
	}
	reshaper, err := reshapes.Stage(handle)
	if err != nil {
		return nil, err
	}
	return &Layer{
		conv:      conv,
		mode:      convtransposetrans,
		trans:     reshaper,
		hiddenmem: vol,
	}, nil
}
func (l *Layer) transposeforward(handle *cudnn.Handler, wspace *cudnn.Malloced, x, y *layers.IO) error {
	err := l.trans.TransformForward(handle, 1, 0, x.T(), l.hiddenmem)
	if err != nil {
		return err
	}
	l.conv.ForwardProp(handle, wspace)
	return nil
}
