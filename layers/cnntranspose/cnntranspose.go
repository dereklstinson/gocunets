package cnntranspose

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/reshapes"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/trainer"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

/*
There is a few ways to test..
1) using transformtensor and increase the size of the tensor. If done correctly you can make it so that every other value has 0.
2) Using batch to shape then shape to batch. This takes at least twice the mem
3) Use resize and then do a resize back prop which back propigates the errors to the source pixel. -- might not work on NCHW

*/

//Layer contains the ops need for ConvTranspose
type Layer struct {
	conv         *cnn.Layer
	trans        *reshapes.Ops
	hiddenmem    *layers.IO
	hiddenmem2   *layers.IO
	mode         convtransposemode
	resizeddims  []int32
	previouss2b  []int32
	s2bwindow    []int32
	thelper      *reshapes.TransFormHelper
	s2bbatchmult int32
	inputlayer   bool
}
type convtransposemode int

const (
	convtransposetrans   = convtransposemode(1)
	convtransposereverse = convtransposemode(2)
)

//Weights exposes the inner convolutional layer that does the convolution method for the transpose
func (l *Layer) Weights() *layers.IO {
	return l.conv.Weights()
}

//Bias returns the bias of the cnn tranpose
func (l *Layer) Bias() *layers.IO {
	return l.conv.Bias()
}

//ForwardProp does the forward propagation of convolution transpose
func (l *Layer) ForwardProp(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	switch l.mode {

	case convtransposetrans:
		return utils.ErrorWrapper("cnntranspose transform forward", l.tranformforward(handle, wspace, x, y))

	case convtransposereverse:
		return utils.ErrorWrapper("cnntranspose reverse forward", l.reverseForwardProp(handle, wspace, x, y))
	}
	return errors.New("ConvTranspose ForwardProp - Shouldn't have reached here")
}

//BackPropData does the back propigation data of convolution transpose
func (l *Layer) BackPropData(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	switch l.mode {

	case convtransposetrans:
		return l.transformBackPropData(handle, wspace, x, y)

	case convtransposereverse:
		return utils.ErrorWrapper("cnntranspose reverse backdata", l.reverseBackPropData(handle, wspace, x, y))

	}
	return errors.New("ConvTranspose BackProp - Shouldn't have reached here")
}

//BackPropFilterData does the back propigation filter and data of convolution transpose
func (l *Layer) BackPropFilterData(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	if l.inputlayer == true {
		switch l.mode {

		case convtransposetrans:
			return l.transformBackPropFilter(handle, wspace, x, y)

		case convtransposereverse:
			return utils.ErrorWrapper("cnntranspose reverse back filterdata", l.reverseBackPropFilter(handle, wspace, x, y))
		}

	} else {

		switch l.mode {

		case convtransposetrans:
			return l.transformBackPropFilterData(handle, wspace, x, y)

		case convtransposereverse:
			return utils.ErrorWrapper("cnntranspose reverse back filterdata", l.reverseBackPropFilterData(handle, wspace, x, y))
		}

	}

	return errors.New("ConvTranspose BackProp - Shouldn't have reached here")
}

//Transform sets up a transform version of cnn transpose
func build(handle *cudnn.Handler,
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
	upscaleddims []int32, //UpscaledDims will be the dims of the input before the convolution
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	mode convtransposemode,
	inputlayer bool,
	managedmem bool) (*Layer, error) {
	conv, err := cnn.SetupDynamic(handle, frmt, dtype, filterdims, convmode, pad, stride, dilation, managedmem)
	if err != nil {
		return nil, err
	}

	vol, err := layers.BuildIO(frmt, dtype, upscaleddims, managedmem)
	if err != nil {
		return nil, err
	}
	reshaper, err := reshapes.Stage(handle)
	if err != nil {
		return nil, err
	}
	return &Layer{
		conv:        conv,
		mode:        mode,
		trans:       reshaper,
		hiddenmem:   vol,
		resizeddims: upscaleddims,
		inputlayer:  inputlayer,
	}, nil
}

//MakeOutputTensor makes the output tensor
func (l *Layer) MakeOutputTensor(handle *cudnn.Handler, input *layers.IO) (*layers.IO, error) {
	switch l.mode {

	case convtransposetrans:
		return l.resizeandTransformoutputdims(handle)

	case convtransposereverse:
		return l.reverseOutput(handle, input)
	}
	return nil, errors.New("ConvTranspose BackProp - Shouldn't have reached here")
}

func (l *Layer) resizeandTransformoutputdims(handle *cudnn.Handler) (*layers.IO, error) {
	return l.conv.MakeOutputTensor(handle, l.hiddenmem)
}

//LoadTrainer loads a trainer into the layer
func (l *Layer) LoadTrainer(handle *cudnn.Handler, wtrainer, btrainer trainer.Trainer) error {
	if l.mode == convtransposereverse {
		//wtrainer.SetRate(-.001)
		//btrainer.SetRate(-.001)
	}
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
