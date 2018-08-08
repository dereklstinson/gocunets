//Package cnn contains structs and methods used to do forward, and backward operations for convolution layers
package cnn

import (
	"errors"
	"image"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor/convolution"
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
	train      trainer.Momentum
	btrain     trainer.Momentum
}

type xtras struct {
	alpha  float64
	alpha2 float64
	beta   float64
}

func appenderror(comment string, err error) error {
	return errors.New(comment + ": " + err.Error())
}

//LoadWValues will load a slice into cuda memory for the Weights.
func (c *Layer) LoadWValues(slice interface{}) error {
	ptr, err := gocudnn.MakeGoPointer(slice)
	if err != nil {
		return err
	}
	return c.w.LoadTValues(ptr)
}

//LoaddWValues will load a slice into cuda memory for the delta Weights.
func (c *Layer) LoaddWValues(slice interface{}) error {
	ptr, err := gocudnn.MakeGoPointer(slice)
	if err != nil {
		return err
	}
	return c.w.LoadDeltaTValues(ptr)
}

//UpdateWeights does the weight update
func (c *Layer) UpdateWeights(handle *gocudnn.Handle, batch int) error {
	err := c.train.UpdateWeights2(handle, c.w, float64(batch))
	if err != nil {
		return appenderror("UpdateWeights-Weights", err)
	}
	err = c.btrain.UpdateWeights2(handle, c.bias, float64(batch))
	if err != nil {
		return appenderror("UpdateWeights-Bias", err)
	}
	return nil
}

//SetupTrainer sets up the momentum trainer
func (c *Layer) SetupTrainer(handle *gocudnn.Handle, decay1, decay2, rate, momentum float64) error {
	c.train = trainer.SetupMomentum(decay1, decay2, rate, momentum)
	err := c.train.LoadGsum(handle, c.w)
	if err != nil {
		return err
	}
	c.btrain = trainer.SetupMomentum(decay1, decay2, rate, momentum)
	err = c.btrain.LoadGsum(handle, c.bias)
	if err != nil {
		return err
	}
	return nil
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

	fmt, dtype, _, err := input.Properties()
	if err != nil {
		return nil, nil, err
	}
	layer, err := LayerSetup(fmt, dtype, filterdims, convmode, pad, stride, dilation, managedmem)
	if err != nil {
		return nil, nil, err
	}
	err = layer.MakeRandomFromFanin(input)
	if err != nil {
		return nil, nil, err
	}
	err = layer.bias.T().SetValues(handle, 0.0)
	output, err := layer.MakeOutputTensor(handle, input, managedmem)
	if err != nil {
		return nil, nil, err
	}
	_, err = layer.SetBestAlgosConsidering(handle, input, output, 0, false)
	if err != nil {
		return nil, nil, err
	}
	return layer, output, nil

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
	fmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	//data gocudnn.DataType,
	pad,
	stride,
	dialation []int32,
	managedmem bool,
) (*Layer, error) {
	conv, err := convolution.Build(convmode, dtype, pad, stride, dialation)
	if err != nil {
		return nil, err
	}
	w, err := layers.BuildIO(fmt, dtype, filterdims, managedmem, false, false)
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
	_, datatype, _, err := w.Properties()
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
		datatype: datatype,
	}, nil
}
func buildbias(weights *layers.IO, managedmem bool) (*layers.IO, error) {
	frmt, dtype, dims, err := weights.Properties()
	if err != nil {
		return nil, err
	}

	//This is a hack. It will only work with only one type of data arangement. This will need a switch for the different types.  {

	outputmaps := dims[0]
	for i := 0; i < len(dims); i++ {

		dims[i] = int32(1)
	}

	dims[1] = outputmaps
	//fmt.Println("Bias Dims:", dims)

	//	}

	return layers.BuildIO(frmt, dtype, dims, managedmem, false, false)
}
func (c *Layer) WeightsFillSlice(input interface{}) error {
	return c.w.T().Memer().FillSlice(input)

}
func (c *Layer) DeltaWeightsFillSlice(input interface{}) error {
	return c.w.DeltaT().Memer().FillSlice(input)
}

//MakeOutputTensor makes the output tensor of the layer
func (c *Layer) MakeOutputTensor(handle *gocudnn.Handle, input *layers.IO, managedmem bool) (*layers.IO, error) {
	dims, err := c.conv.OutputDim(input.T(), c.w.T())
	if err != nil {
		return nil, err
	}
	fmt, dtype, _, err := c.w.Properties()
	if err != nil {
		return nil, err
	}
	output, err := layers.BuildIO(fmt, dtype, dims, managedmem, false, false)
	if err != nil {
		return nil, err
	}
	/*	err = output.T().SetValues(handle, 1.0)
		if err != nil {
			return nil, err
		}
	*/
	return output, nil
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
func (c *Layer) SaveImagesToFile(dir string) error {
	return c.w.SaveImagesToFile(dir)
}
func (c *Layer) WeightImgs() ([][]image.Image, [][]image.Image, error) {
	return c.w.Images()
}
func (c *Layer) BiasImgs() ([][]image.Image, [][]image.Image, error) {
	return c.bias.Images()
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

//SetBestAlgosConsidering this method will set the best algos for the fwd, bwddata, and bwdfilter algos. and return the workspace size along with an error
//if an error is found the function will not set any values,
//Here are some simple rules to the function
//if fastest is marked true. Then it will find the fastest algo no mater what worksize is.
//if fastest is set to false. It will check if wspace is greater than zero then it will set the algos to the fastest algo considering the workspace size, and return the largest wspacesize in all the algos
//else it will find and set the fastest algos with no workspace size and return 0
func (c *Layer) SetBestAlgosConsidering(handle *gocudnn.Handle, x, y *layers.IO, wspacelimit int, fastest bool) (gocudnn.SizeT, error) {
	return c.conv.SetBestAlgosConsidering(handle, x.T(), y.T(), c.w.T(), wspacelimit, fastest)
}

/*
//Build builds the CNNlayer with the descriptors that are in it.  Returns how much memory is left to use
func (c *Layer) Build() error {
	var err error
	c.dw, err = gocudnn.Malloc(c.size)
	if err != nil {
		return err
	}
	c.w, err = gocudnn.Malloc(c.size)
	if err != nil {
		return err
	}
	return nil

}

*/
