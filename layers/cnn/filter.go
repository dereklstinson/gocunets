//Package cnn contains structs and methods used to do forward, and backward operations for convolution layers
package cnn

import (
	"encoding/json"
	"errors"
	"os"

	"github.com/dereklstinson/GoCuNets/gocudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/trainer"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

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
	datatype   gocudnn.DataType
	train      trainer.Trainer
	btrain     trainer.Trainer
	inputdims  []int32
	outputdims []int32
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

//SaveJSON saves the weights to json
func (c *Layer) SaveJSON(folder, name string) error {
	var save TempSave
	w, err := c.w.Info()
	if err != nil {
		return err
	}
	b, err := c.bias.Info()
	if err != nil {
		return err
	}
	save.Bias = b
	save.Weights = w
	marshed, err := json.Marshal(save)
	if err != nil {
		return err
	}
	dir := folder + "/"
	err = os.MkdirAll(dir, os.ModePerm)
	if err != nil {
		return err
	}
	newfile, err := os.Create(dir + name + ".json")
	if err != nil {
		return err
	}
	_, err = newfile.Write(marshed)
	if err != nil {

		return err
	}
	defer newfile.Close()

	return nil
}

//UpdateWeights does the weight update
func (c *Layer) UpdateWeights(handle gocudnn.Handler, batch int) error {
	err := c.btrain.UpdateWeights(handle, c.bias, batch)
	if err != nil {
		return err
	}
	return c.train.UpdateWeights(handle, c.w, batch)
}

//LoadTrainer sets up the momentum trainer
func (c *Layer) LoadTrainer(handle gocudnn.Handler, trainerweights, trainerbias trainer.Trainer) error {
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

//LayerSetupPredefinedWeightsDefault sets up a default layer with the weights already been made
func LayerSetupPredefinedWeightsDefault(
	handle *gocudnn.Handle,
	input *layers.IO,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	managedmem bool,
	weights interface{},
	bias interface{},
) (*Layer, *layers.IO, error) {

	frmt, dtype, _, err := input.Properties()
	if err != nil {
		return nil, nil, err
	}
	layer, err := LayerSetup(frmt, dtype, filterdims, convmode, pad, stride, dilation, managedmem)
	if err != nil {
		return nil, nil, err
	}
	err = layer.LoadWValues(weights)
	if err != nil {
		return nil, nil, err
	}
	err = layer.LoadBiasValues(bias)
	output, err := layer.MakeOutputTensor(handle, input)
	if err != nil {
		return nil, nil, err
	}
	_, err = layer.SetBestAlgosConsidering(handle, input, output, 0, false)
	if err != nil {
		return nil, nil, err
	}
	return layer, output, nil

}

//AIOLayerSetupDefault builds a layer based on the input, and other values passed.
//It will choose the propigation algos with the consideration of not wanting any workspace.
//It will also build and pass the outputlayer.
func AIOLayerSetupDefault(
	handle *gocudnn.Handle,
	input *layers.IO,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	managedmem bool,
) (*Layer, *layers.IO, error) {

	frmt, dtype, _, err := input.Properties()
	if err != nil {
		return nil, nil, err
	}
	layer, err := LayerSetup(frmt, dtype, filterdims, convmode, pad, stride, dilation, managedmem)
	if err != nil {
		return nil, nil, err
	}
	err = layer.MakeRandomFromFanin(input)
	if err != nil {
		return nil, nil, err
	}
	err = layer.bias.T().SetValues(handle, 0.0)
	if err != nil {
		return nil, nil, err
	}
	output, err := layer.MakeOutputTensor(handle, input)
	if err != nil {

		return nil, nil, err
	}
	_, err = layer.SetBestAlgosConsidering(handle, input, output, 0, false)
	if err != nil {
		return nil, nil, err
	}
	return layer, output, nil

}

//LayerSetupV2 handles setting up the layer without knowing the input
func LayerSetupV2(handle *gocudnn.Handle,
	frmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
	inputdims []int32,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	workspace int,
	fastest bool,
	managedmem bool) (*Layer, error) {

	layer, err := LayerSetup(frmt, dtype, filterdims, convmode, pad, stride, dilation, managedmem)
	if err != nil {
		return nil, err
	}
	err = layer.MakeRandomFromFaninDims(inputdims)
	if err != nil {
		return nil, err
	}
	layer.inputdims = inputdims
	err = layer.bias.T().SetValues(handle, 0.0)
	if err != nil {
		return nil, err
	}
	outputdims := find4doutputdims(inputdims, filterdims, pad, stride, dilation, frmt)
	layer.outputdims = outputdims
	_, err = layer.SetBestAlgosConsideringDims4d(handle, inputdims, outputdims, filterdims, workspace, fastest)
	if err != nil {
		return nil, err
	}
	return layer, nil
}

//AIOLayerSetupDefaultNoOut takes the input and figures out all the configurations with the default being fastest but no workspace
func AIOLayerSetupDefaultNoOut(
	handle *gocudnn.Handle,
	input *layers.IO,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	managedmem bool,
) (*Layer, error) {

	frmt, dtype, inputdims, err := input.Properties()
	if err != nil {
		return nil, err
	}
	layer, err := LayerSetup(frmt, dtype, filterdims, convmode, pad, stride, dilation, managedmem)
	if err != nil {
		return nil, err
	}
	layer.inputdims = inputdims
	frmt, _, wdims, _ := layer.w.Properties()
	outputdims := find4doutputdims(inputdims, wdims, pad, stride, dilation, frmt)
	layer.outputdims = outputdims
	err = layer.MakeRandomFromFanin(input)
	if err != nil {
		return nil, err
	}

	err = layer.bias.T().SetValues(handle, 0.0)
	if err != nil {
		return nil, err
	}
	_, err = layer.SetBestAlgosConsideringDims4d(handle, inputdims, outputdims, wdims, 0, false)
	if err != nil {
		return nil, err
	}
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
func LayerSetup(
	frmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
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

	alpha := 1.0
	alpha2 := 1.0
	beta := 0.0
	beta2 := 1.0
	return &Layer{
		size: sizeinbytes,
		conv: conv,

		w:    w,
		bias: bias,
		fwd: xtras{
			alpha:  alpha,
			alpha2: alpha2,
			beta:   beta,
		},
		bwdd: xtras{
			alpha:  alpha,
			alpha2: alpha2,
			beta:   beta,
		},
		bwdf: xtras{
			alpha:  alpha,
			alpha2: alpha2,
			beta:   beta2,
		},
		datatype: dtype,
	}, nil
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

//ForwardProp performs the ForwardProp
func (c *Layer) ForwardProp(handle *gocudnn.Handle, wspace gocudnn.Memer, x, y *layers.IO) error {
	//	fmt.Println("alpha: ", c.fwd.alpha, ",   beta: ", c.fwd.beta)
	err := c.conv.FwdProp(handle, c.fwd.alpha,
		x.T(),
		c.w.T(),
		wspace,
		c.fwd.beta,
		y.T(),
	)
	if err != nil {
		return err
	}
	/*
		_, _, outputdims, _ := y.Properties()
		fmt.Println("Output Dims: ", outputdims)
		_, _, biasdims, _ := c.bias.Properties()
		fmt.Println("Bias Dims: ", biasdims)
	*/
	return y.T().AddTo(handle, c.bias.T(), 1.0, 1.0)

}

/*
//ForwardBiasActivation does the forward bias activation cudnn algorithm
func (c *Layer) ForwardBiasActivation(handle *gocudnn.Handle, x *layers.IO, wpsace gocudnn.Memer, z *layers.IO, aD *gocudnn.ActivationD, y *layers.IO) error {
	return c.cfuncs.Fwd.ConvolutionBiasActivationForward(
		handle,
		c.fwd.alpha,
		x.Tensor().TD(),
		x.Mem(),
		c.w.DTensor().FD(),
		c.w.DMem(),
		c.cD,
		c.fwdAlgo,
		wpsace,
		c.fwd.alpha2,
		z.Tensor().TD(),
		z.DMem(),
		c.bias.DTensor().TD(),
		c.bias.Mem(),
		aD,
		y.Tensor().TD(),
		y.Mem())
}
*/

//BackProp does the backprop for the data and the filter
func (c *Layer) BackProp(handle *gocudnn.Handle, wspace gocudnn.Memer, x, y *layers.IO) error {
	var err error
	if x.IsInput() == true {
		return c.backpropfilter(handle, wspace, x, y)
	}
	err = c.backpropdata(handle, wspace, x, y)
	if err != nil {
		return err
	}

	return c.backpropfilter(handle, wspace, x, y)
}

//BackPropData performs the BackPropData
func (c *Layer) backpropdata(handle *gocudnn.Handle, wspace gocudnn.Memer, x, y *layers.IO) error {
	return c.conv.BwdPropData(
		handle,
		c.bwdd.alpha,
		c.w.T(),
		y.DeltaT(),
		wspace,
		c.bwdd.beta,
		x.DeltaT(),
	)

}

//BackPropFilter does the backward propagation for the filter You will pass a handle workspace memory x,dy layer.io
func (c *Layer) backpropfilter(handle *gocudnn.Handle, wspace gocudnn.Memer, x, y *layers.IO) error {
	err := c.conv.BwdPropFilt(
		handle,
		c.bwdf.alpha,
		x.T(),
		y.DeltaT(),
		wspace,
		c.bwdf.beta,
		c.w.DeltaT())
	if err != nil {
		return err
	}

	return c.conv.BwdBias(
		handle,
		c.bwdf.alpha,
		y.DeltaT(),
		c.bwdf.beta,
		c.bias.DeltaT())

}

func find4doutputdims(x, w, padding, stride, dilation []int32, frmt gocudnn.TensorFormat) []int32 {
	var flag gocudnn.TensorFormatFlag
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
