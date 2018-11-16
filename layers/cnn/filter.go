//Package cnn contains structs and methods used to do forward, and backward operations for convolution layers
package cnn

import (
	"encoding/json"
	"errors"
	"io"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/trainer"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

const alphadefault = 1.0
const beta1default = 0.0
const beta2default = 1.0

//Layer is a struct that holds  filter, bias and convolution descriptors.
//The memory for w, dw, bias, dbias. The algos for forward, backward (data, filter) and the scalars for those algos. 1
type Layer struct {
	conv       *convolution.Ops
	w          *layers.IO
	bias       *layers.IO
	size       gocudnn.SizeT
	wspacesize gocudnn.SizeT
	fwd        xtras
	bwdd       xtras
	bwdf       xtras
	datatype   cudnn.DataType
	train      trainer.Trainer
	btrain     trainer.Trainer
	pad        []int32
	dilation   []int32
	stride     []int32
	l1b        float32
	l2b        float32
	l1w        float32
	l2w        float32
}

//TempSave is a temporary struct for saving.  It will most likely be changed
type TempSave struct {
	Weights layers.Info `json:"Weights"`
	Bias    layers.Info `json:"Bias"`
}
type xtras struct {
	alpha  float64
	alpha2 float64
	beta   float64
}

func appenderror(comment string, err error) error {
	return errors.New(comment + ": " + err.Error())
}

//Encode saves the weights to json
func (c *Layer) Encode(w io.Writer) error {
	var save TempSave
	weights, err := c.w.Info()
	if err != nil {
		return err
	}
	b, err := c.bias.Info()
	if err != nil {
		return err
	}
	save.Bias = b
	save.Weights = weights
	marshed, err := json.Marshal(save)
	if err != nil {
		return err
	}
	_, err = w.Write(marshed)
	if err != nil {

		return err
	}
	return nil
}

//UpdateWeights does the weight update
func (c *Layer) UpdateWeights(handle *cudnn.Handler, batch int) error {
	err := c.btrain.UpdateWeights(handle, c.bias, batch)
	if err != nil {
		return err
	}
	c.l1b, c.l2b, err = c.btrain.L1L2Loss()
	if err != nil {
		return err
	}
	c.train.UpdateWeights(handle, c.w, batch)
	c.l1w, c.l2w, err = c.train.L1L2Loss()
	if err != nil {
		return err
	}
	return nil
}

//L1L2Loss will return the L1 loss and L2 loss for the layer
func (c *Layer) L1L2Loss() (L1 float32, L2 float32) {
	return c.l1b + c.l1w, c.l2b + c.l2w
}

//LoadTrainer sets up the momentum trainer
func (c *Layer) LoadTrainer(handle *cudnn.Handler, trainerweights, trainerbias trainer.Trainer) error {
	var err error
	c.train = trainerweights
	err = trainer.CreateTrainingMem(handle, c.train, c.w)
	if err != nil {
		return err
	}
	c.btrain = trainerbias
	err = trainer.CreateTrainingMem(handle, c.btrain, c.bias)
	if err != nil {
		return err
	}
	return nil
}

//Bias returns the Bias
func (c *Layer) Bias() *layers.IO {
	return c.bias
}

//Weights returns the weights
func (c *Layer) Weights() *layers.IO {
	return c.w
}

//LoadLayerStatic sets up a default layer with the weights already been made
func LoadLayerStatic(
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
	handle *cudnn.Handler,
	inputdims []int32,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	managedmem bool,
	fastest bool,
	wspacesize int,
	weights interface{},
	bias interface{},
) (*Layer, error) {

	layer, err := layersetup(frmt, dtype, filterdims, convmode, pad, stride, dilation, managedmem)
	if err != nil {
		return nil, err
	}
	err = layer.LoadWValues(weights)
	if err != nil {
		return nil, err
	}
	err = layer.LoadBiasValues(bias)

	if err != nil {
		return nil, err
	}
	_, _, wdims, err := layer.w.Properties()
	if err != nil {
		return nil, err
	}
	outputdims := find4doutputdims(inputdims, wdims, pad, stride, dilation, frmt)
	layer.wspacesize, err = layer.SetBestAlgosConsideringDims4d(handle, inputdims, outputdims, wdims, wspacesize, fastest)
	if err != nil {
		return nil, err
	}
	return layer, nil
}

//LoadLayerDynamic sets up a default layer with the weights already been made
func LoadLayerDynamic(
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
	handle *gocudnn.Handle,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	managedmem bool,
	weights interface{},
	bias interface{},
) (*Layer, error) {

	layer, err := layersetup(frmt, dtype, filterdims, convmode, pad, stride, dilation, managedmem)
	if err != nil {
		return nil, err
	}
	err = layer.LoadWValues(weights)
	if err != nil {
		return nil, err
	}
	err = layer.LoadBiasValues(bias)
	return layer, nil

}

//OutputDims will return the dims for the output
func (c *Layer) OutputDims(inputdims []int32) []int32 {
	if len(inputdims) != 4 {
		return nil
	}
	frmt, _, dims, err := c.w.Properties()
	if err != nil {
		return nil
	}

	return find4doutputdims(inputdims, dims, c.pad, c.stride, c.dilation, frmt)

}

//SetupDynamic sets up the speed of the fwd and bwd algos dynamically.  guessinputdims is really for setting up the random weights.
func SetupDynamic(handle *cudnn.Handler,
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
	guessinputdims []int32,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	managedmem bool) (*Layer, error) {
	layer, err := layersetup(frmt, dtype, filterdims, convmode, pad, stride, dilation, managedmem)
	if err != nil {
		return nil, err
	}
	err = layer.MakeRandomFromFaninDims(guessinputdims)
	if err != nil {
		return nil, err
	}

	err = layer.bias.T().SetValues(handle, 0.001)
	if err != nil {
		return nil, err
	}

	layer.pad, layer.stride, layer.dilation = pad, stride, dilation
	return layer, nil
}

//SetUpStatic sets up the layer and decides on the layers fwd and bwd algos on first build. Good for static sized inputs.
func SetUpStatic(handle *cudnn.Handler,
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
	inputdims []int32,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	workspace int,
	fastest bool,
	managedmem bool) (*Layer, error) {

	layer, err := layersetup(frmt, dtype, filterdims, convmode, pad, stride, dilation, managedmem)
	if err != nil {
		return nil, err
	}
	err = layer.MakeRandomFromFaninDims(inputdims)
	if err != nil {
		return nil, err
	}

	err = layer.bias.T().SetValues(handle, 0.0)
	if err != nil {
		return nil, err
	}
	outputdims := find4doutputdims(inputdims, filterdims, pad, stride, dilation, frmt)

	_, err = layer.SetBestAlgosConsideringDims4d(handle, inputdims, outputdims, filterdims, workspace, fastest)
	if err != nil {
		return nil, err
	}
	layer.pad, layer.stride, layer.dilation = pad, stride, dilation
	return layer, nil
}

//MakeRandomFromFaninDims does what it says it will make the weights random considering the fanin
func (c *Layer) MakeRandomFromFaninDims(dims []int32) error {

	if len(dims) < 5 {
		fanin := float64(dims[1] * dims[2] * dims[3])
		err := c.w.T().SetRandom(0, 1.0, fanin)
		if err != nil {
			return err
		}

	}
	if len(dims) > 4 {
		return errors.New("Not Available yet")
	}

	return nil
}

//MakeRandomFromFanin does what it says it will make the weights random considering the fanin
func (c *Layer) MakeRandomFromFanin(input *layers.IO) error {
	_, _, dims, err := input.Properties()
	if err != nil {
		return err
	}
	if len(dims) < 5 {
		fanin := float64(dims[1] * dims[2] * dims[3])
		err := c.w.T().SetRandom(0, 1.0, fanin)
		if err != nil {
			return err
		}

	}
	if len(dims) > 4 {
		return errors.New("Not Available yet")
	}

	return nil
}

//LayerSetup sets up the cnn layer to be built. But doesn't build it yet.
func layersetup(
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dialation []int32,
	managedmem bool,
) (*Layer, error) {
	conv, err := convolution.StageOperation(convmode, dtype, pad, stride, dialation)
	if err != nil {
		return nil, err
	}
	w, err := layers.BuildIO(frmt, dtype, filterdims, managedmem)
	if err != nil {
		return nil, err
	}

	sizeinbytes, err := w.T().Size()
	if err != nil {
		return nil, err
	}
	bias, err := buildbias(w, managedmem)
	if err != nil {
		return nil, err
	}

	return &Layer{
		size: sizeinbytes,
		conv: conv,

		w:    w,
		bias: bias,
		fwd: xtras{
			alpha:  alphadefault,
			alpha2: alphadefault,
			beta:   beta1default,
		},
		bwdd: xtras{
			alpha:  alphadefault,
			alpha2: alphadefault,
			beta:   beta1default,
		},
		bwdf: xtras{
			alpha:  alphadefault,
			alpha2: alphadefault,
			beta:   beta2default,
		},
		datatype: dtype,
	}, nil
}

//SetFwdScalars sets the alpha and beta scalars, the defaults are alpha, alpha2 =1, 1, beta=0 and are initialized in the function FilterSetup
func (c *Layer) SetFwdScalars(alpha, alpha2, beta float64) {
	c.fwd.alpha, c.fwd.alpha2, c.fwd.beta = alpha, alpha2, beta
}

//SetBwdDataScalars sets the alpha and beta scalars, the defaults are alpha, alpha2 =1, 1, beta=0 and are initialized in the function FilterSetup
func (c *Layer) SetBwdDataScalars(alpha, alpha2, beta float64) {
	c.bwdd.alpha, c.bwdd.alpha2, c.bwdd.beta = alpha, alpha2, beta
}

//SetBwdFilterScalars sets the alpha and beta scalars, the defaults are alpha, alpha2 =1, 1, beta=1 and are initialized in the function FilterSetup
func (c *Layer) SetBwdFilterScalars(alpha, alpha2, beta float64) {
	c.bwdf.alpha, c.bwdf.alpha2, c.bwdf.beta = alpha, alpha2, beta
}

func find4doutputdims(x, w, padding, stride, dilation []int32, frmt cudnn.TensorFormat) []int32 {
	var flag cudnn.TensorFormatFlag
	if frmt == flag.NCHW() {
		return find4doutputdims4dNCHW(x, w, padding, stride, dilation)
	}
	return find4doutputdims4dNHWC(x, w, padding, stride, dilation)
}
func find4doutputdims4dNCHW(x, w, padding, stride, dilation []int32) []int32 {
	out := make([]int32, len(x))
	out[0] = x[0]
	out[1] = w[0]
	out[2] = findoutputdim(x[2], w[2], stride[0], padding[0], dilation[0])
	out[3] = findoutputdim(x[3], w[3], stride[1], padding[1], dilation[1])
	return out
}
func find4doutputdims4dNHWC(x, w, padding, stride, dilation []int32) []int32 {
	out := make([]int32, len(x))
	out[0] = x[0]
	out[1] = findoutputdim(x[1], w[1], stride[0], padding[0], dilation[0])
	out[2] = findoutputdim(x[2], w[2], stride[1], padding[1], dilation[1])
	out[3] = w[0]

	return out
}

/* for NCHW filter is KCHW
(
 K represents the number of output feature maps,
 C the number of input feature maps,
 R the number of rows per filter,
 S the number of columns per filter.)
for NHWC filter is KRSC
 K represents the number of output feature maps,
 R the number of rows per filter,
 S the number of columns per filter.
 C the number of input feature maps)
*/
func findoutputdim(x, w, s, p, d int32) int32 {
	return 1 + (x+2*p-(((w-1)*d)+1))/s
}
func buildbias(weights *layers.IO, managedmem bool) (*layers.IO, error) {
	frmt, dtype, dims, err := weights.Properties()
	if err != nil {
		return nil, err
	}
	outputmaps := dims[0]
	for i := 0; i < len(dims); i++ {

		dims[i] = int32(1)
	}
	dims[1] = outputmaps
	return layers.BuildIO(frmt, dtype, dims, managedmem)
}
