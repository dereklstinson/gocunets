//Package cnn contains structs and methods used to do forward, and backward operations for convolution layers
package cnn

import (
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

//update weights updates the weights of the neurons
func (c *Layer) UpdateWeights(handle *gocudnn.Handle) error {
	err := c.train.UpdateWeights(handle, c.w)
	if err != nil {
		return err
	}
	return c.btrain.UpdateWeights(handle, c.bias)
}

//SetupTrainer sets up the momentum trainer
func (c *Layer) SetupTrainer(handle *gocudnn.Handle, decay1, decay2, rate, momentum float64) error {
	c.train = trainer.SetupMomentum(decay1, decay2, rate, momentum)
	c.btrain = trainer.SetupMomentum(decay1, decay2, rate, momentum)
	err := c.btrain.LoadGsum(handle, c.bias)
	if err != nil {
		return err
	}
	return c.train.LoadGsum(handle, c.w)
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
	output, err := layer.MakeOutputTensor(input, managedmem)
	if err != nil {
		return nil, nil, err
	}
	_, err = layer.SetBestAlgosConsidering(handle, input, output, 0, false)
	if err != nil {
		return nil, nil, err
	}
	return layer, output, nil

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
	w, err := layers.BuildIO(fmt, dtype, filterdims, managedmem)
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
	fmt, dtype, dims, err := weights.Properties()
	if err != nil {
		return nil, err
	}
	for i := 1; i < len(dims); i++ {
		dims[i] = int32(1)
	}
	return layers.BuildIO(fmt, dtype, dims, managedmem)
}

//MakeOutputTensor makes the output tensor of the layer
func (c *Layer) MakeOutputTensor(input *layers.IO, managedmem bool) (*layers.IO, error) {
	dims, err := c.conv.OutputDim(input.T(), c.w.T())
	if err != nil {
		return nil, err
	}
	fmt, dtype, _, err := c.w.Properties()
	return layers.BuildIO(fmt, dtype, dims, managedmem)
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
	return y.T().AddTo(handle, c.bias.T(), c.fwd.alpha, c.fwd.beta)

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
