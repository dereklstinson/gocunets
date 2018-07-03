//Package cnn contains structs and methods used to do forward, and backward operations for convolution layers
package cnn

import (
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is a struct that holds  filter, bias and convolution descriptors.
//The memory for w, dw, bias, dbias. The algos for forward, backward (data, filter) and the scalars for those algos. 1
type Layer struct {
	cfuncs      gocudnn.ConvolutionFuncs
	tfuncs      gocudnn.TensorFuncs
	cD          *gocudnn.ConvolutionD
	w           *layers.IO
	bias        *layers.IO
	size        gocudnn.SizeT
	fwdAlgo     gocudnn.ConvFwdAlgo
	bwdAlgodata gocudnn.ConvBwdDataAlgo
	bwdAlgoFilt gocudnn.ConvBwdFiltAlgo
	fwd         xtras
	bwdd        xtras
	bwdf        xtras
	datatype    gocudnn.DataType
}

type xtras struct {
	alpha  gocudnn.CScalar
	alpha2 gocudnn.CScalar
	beta   gocudnn.CScalar
}

//LayerSetup sets up the cnn layer to be built. But doesn't build it yet.
func LayerSetup(input *gocudnn.TensorD,
	w *layers.IO,
	cD *gocudnn.ConvolutionD,
	fwdAlgo gocudnn.ConvFwdAlgo,
	bwdAlgodata gocudnn.ConvBwdDataAlgo,
	bwdAlgoFilt gocudnn.ConvBwdFiltAlgo,
	sizeinbytes gocudnn.SizeT) (*Layer, error) {
	_, datatype, _, err := w.Properties()
	if err != nil {
		return nil, err
	}

	alpha := gocudnn.FindScalar(datatype, 1)
	alpha2 := gocudnn.FindScalar(datatype, 1)
	beta := gocudnn.FindScalar(datatype, 0)
	return &Layer{
		size:        sizeinbytes,
		cD:          cD,
		w:           w,
		fwdAlgo:     fwdAlgo,
		bwdAlgodata: bwdAlgodata,
		bwdAlgoFilt: bwdAlgoFilt,
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
			beta:   beta,
		},
		datatype: datatype,
	}, nil
}

//SetFwdScalars sets the alpha and beta scalars, the defaults are alpha, alpha2 =1, 1, beta=0 and are initialized in the function FilterSetup
func (c *Layer) SetFwdScalars(alpha, alpha2, beta gocudnn.CScalar) {
	c.fwd.alpha, c.fwd.alpha2, c.fwd.beta = alpha, alpha2, beta
}

//SetBwdDataScalars sets the alpha and beta scalars, the defaults are alpha, alpha2 =1, 1, beta=0 and are initialized in the function FilterSetup
func (c *Layer) SetBwdDataScalars(alpha, alpha2, beta gocudnn.CScalar) {
	c.bwdd.alpha, c.bwdd.alpha2, c.bwdd.beta = alpha, alpha2, beta
}

//SetBwdFilterScalars sets the alpha and beta scalars, the defaults are alpha, alpha2 =1, 1, beta=0 and are initialized in the function FilterSetup
func (c *Layer) SetBwdFilterScalars(alpha, alpha2, beta gocudnn.CScalar) {
	c.bwdf.alpha, c.bwdf.alpha2, c.bwdf.beta = alpha, alpha2, beta
}

//ForwardProp performs the ForwardProp
func (c *Layer) ForwardProp(handle *gocudnn.Handle, wspace gocudnn.Memer, x, y *layers.IO) error {
	err := c.cfuncs.Fwd.ConvolutionForward(handle, c.fwd.alpha,
		x.Tensor().TD(),
		x.Mem(),
		c.w.Tensor().FD(),
		c.w.Mem(),
		c.cD,
		c.fwdAlgo,
		wspace,
		c.fwd.beta,
		y.Tensor().TD(),
		y.Mem())
	if err != nil {
		return err
	}
	return c.tfuncs.AddTensor(handle, c.datatype, c.fwd.alpha, c.bias.Tensor().TD(), c.bias.Mem(), c.fwd.beta, y.Tensor().TD(), y.Mem())

}

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

//BackPropData performs the BackPropData
func (c *Layer) BackPropData(handle *gocudnn.Handle, wspace gocudnn.Memer, x, y *layers.IO) error {
	return c.cfuncs.Bwd.ConvolutionBackwardData(
		handle,
		c.bwdd.alpha,
		c.w.Tensor().FD(),
		c.w.Mem(),
		y.DTensor().TD(),
		y.DMem(),
		c.cD,
		c.bwdAlgodata,
		wspace,
		c.bwdd.beta,
		x.DTensor().TD(),
		x.DMem())

}

//BackPropFilter does the backward propagation for the filter You will pass a handle workspace memory x,dy layer.io
func (c *Layer) BackPropFilter(handle *gocudnn.Handle, wspace gocudnn.Memer, x, y *layers.IO) error {
	err := c.cfuncs.Bwd.ConvolutionBackwardFilter(
		handle,
		c.bwdf.alpha,
		x.Tensor().TD(),
		x.Mem(),
		y.DTensor().TD(),
		y.DMem(),
		c.cD,
		c.bwdAlgoFilt,
		wspace,
		c.bwdf.beta,
		c.w.DTensor().FD(),
		c.w.DMem())
	if err != nil {
		return err
	}

	return c.cfuncs.Bwd.ConvolutionBackwardBias(handle,
		c.bwdf.alpha,
		y.DTensor().TD(),
		y.DMem(),
		c.bwdf.beta,
		c.bias.DTensor().TD(),
		c.bias.DMem())

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
