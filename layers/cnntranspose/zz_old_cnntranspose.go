package cnntranspose

/*

This is old
import (
	"errors"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/layers"
	"github.com/dereklstinson/gocunets/layers/cnn"
	"github.com/dereklstinson/gocunets/trainer"
	"github.com/dereklstinson/gocunets/utils"
	gocudnn "github.com/dereklstinson/gocudnn"
)


//There is a few ways to test..
//1) using transformtensor and increase the size of the tensor. If done correctly you can make it so that every other value has 0.
//2) Using batch to shape then shape to batch. This takes at least twice the mem
//3) Use resize and then do a resize back prop which back propigates the errors to the source pixel. -- might not work on NCHW



//Layer contains the ops need for ConvTranspose
type Layer struct {
	conv        *cnn.Layer
	mode        convtransposemode
	resizeddims []int32
	previouss2b []int32
	s2bwindow   []int32
}
type convtransposemode int

const (
	convtransposereverse = convtransposemode(2)
)

//Weights exposes the inner convolutional layer that does the convolution method for the transpose
func (l *Layer) Weights() *layers.Tensor {
	return l.conv.Weights()
}

//DeltaWeights returns the delta weights
func (l *Layer) DeltaWeights() *layers.Tensor {
	return l.conv.DeltaWeights()
}

//Bias returns the bias of the cnn tranpose
func (l *Layer) Bias() *layers.Tensor {
	return l.conv.Bias()
}

//SetMathType sets the mathtype
func (l *Layer) SetMathType(mtype gocudnn.MathType) error {
	return l.conv.SetMathType(mtype)
}

//ForwardProp does the forward propagation of convolution transpose
func (l *Layer) ForwardProp(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.Tensor) error {
	switch l.mode {

	case convtransposereverse:
		return utils.ErrorWrapper("cnntranspose reverse forward", l.reverseForwardProp(handle, wspace, x, y))
	}
	return errors.New("ConvTranspose ForwardProp - Shouldn't have reached here")
}

//BackPropData does the back propigation data of convolution transpose
func (l *Layer) BackPropData(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.Tensor) error {
	switch l.mode {

	case convtransposereverse:
		return utils.ErrorWrapper("cnntranspose reverse backdata", l.reverseBackPropData(handle, wspace, x, y))

	}
	return errors.New("ConvTranspose BackProp - Shouldn't have reached here")
}

//BackPropFilter performs the backprop for the filter.
func (l *Layer) BackPropFilter(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.Tensor) error {
	return utils.ErrorWrapper("cnntranspose reverse back filterdata", l.reverseBackPropFilter(handle, wspace, x, y))
}

//BackPropFilterData does the back propigation filter and data of convolution transpose
func (l *Layer) BackPropFilterData(handle *cudnn.Handler, wspacedata, wspacefilter *nvidia.Malloced, x, dx, dy *layers.Tensor) error {
	if dx == nil {
		return utils.ErrorWrapper("cnntranspose reverse back filterdata dx is nil", l.reverseBackPropFilter(handle, wspacefilter, x, dy))
	}
	return utils.ErrorWrapper("cnntranspose reverse back filterdata", l.reverseBackPropFilterData(handle, wspacedata, wspacefilter, x, dx, dy))
}

//Transform sets up a transform version of cnn transpose
func build(handle *cudnn.Handler,
	frmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
	mtype gocudnn.MathType,
	groupcount int32,
	upscaleddims []int32, //UpscaledDims will be the dims of the input before the convolution
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	mode convtransposemode,
	seed uint64) (*Layer, error) {
	conv, err := cnn.Setup(handle, frmt, dtype, mtype, groupcount, filterdims, convmode, pad, stride, dilation, seed)
	if err != nil {
		return nil, err
	}

	return &Layer{
		conv:        conv,
		mode:        mode,
		resizeddims: upscaleddims,
	}, nil
}
func SetupBasic(handle *cudnn.Handler,
	frmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
	mtype gocudnn.MathType,
	groupcount int32,
	w, dw, b, db *layers.Tensor,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32) (*Layer, error) {
	x, err := cnn.SetupBasic(handle, frmt, dtype, mtype, groupcount, w, dw, b, db, convmode, pad, stride, dilation)
	if err != nil {
		return nil, err
	}
	return &Layer{
		conv: x,
		mode: convtransposereverse,
	}, nil
}

//MakeRandom makes the weights random
func (l *Layer) MakeRandom(inputdims []int32) error {
	return l.conv.MakeRandom(inputdims)
}

//FindOutputDims will return the output dims of the reverse convolution
func (l *Layer) FindOutputDims(input *layers.Tensor) ([]int32, error) {
	switch l.mode {
	case convtransposereverse:
		return l.conv.FindReverseOutputDims(input)
	}
	return nil, errors.New("Unsupported Mode")
}

//MakeOutputTensor makes the output tensor
func (l *Layer) MakeOutputTensor(handle *cudnn.Handler, input *layers.Tensor) (*layers.Tensor, error) {
	switch l.mode {

	case convtransposereverse:
		return l.reverseOutput(handle, input)
	}
	return nil, errors.New("ConvTranspose BackProp - Shouldn't have reached here")
}


////MakeOutputTensorInference makes the output tensor for inference only
//func (l *Layer) MakeOutputTensorInference(handle *cudnn.Handler, input *layers.Tensor) (*layers.Tensor, error) {
//	switch l.mode {
//
//	case convtransposereverse:
//		return l.reverseOutputInference(handle, input)
//	}
//	return nil, errors.New("ConvTranspose BackProp - Shouldn't have reached here")
//}


//LoadTrainer loads a trainer into the layer
func (l *Layer) LoadTrainer(handle *cudnn.Handler, wtrainer, btrainer trainer.Trainer) error {

	return l.conv.LoadTrainer(handle, wtrainer, btrainer)
}

//L1L2Loss returns the L1 and L2 loss for the layer
func (l *Layer) L1L2Loss() (L1, L2 float32) {
	return l.conv.L1L2Loss()
}

//UpdateWeights updates the weights in the layer
func (l *Layer) UpdateWeights(handle *cudnn.Handler, batch int) error {

	return l.conv.UpdateWeights(handle, batch)
}
*/
