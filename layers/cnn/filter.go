//Package cnn contains structs and methods used to do forward, and backward operations for convolution layers
package cnn

import (
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is a struct that holds  filter, bias and convolution descriptors.
//The memory for w, dw, bias, dbias. The algos for forward, backward (data, filter) and the scalars for those algos.
type Layer struct {
	cD          *gocudnn.ConvolutionD
	wD          *gocudnn.FilterD
	dwD         *gocudnn.FilterD
	biasD       *gocudnn.TensorD
	w           gocudnn.Memer
	dw          gocudnn.Memer
	bias        gocudnn.Memer
	dbias       gocudnn.Memer
	size        gocudnn.SizeT
	fwdAlgo     gocudnn.ConvFwdAlgo
	bwdAlgodata gocudnn.ConvBwdDataAlgo
	bwdAlgoFilt gocudnn.ConvBwdFiltAlgo
	fwd         xtras
	bwdd        xtras
	bwdf        xtras
}

type xtras struct {
	alpha  gocudnn.CScalar
	alpha2 gocudnn.CScalar
	beta   gocudnn.CScalar
}

//LayerSetup sets up the cnn layer to be built. But doesn't build it yet.
func LayerSetup(input *gocudnn.TensorD,
	wD *gocudnn.FilterD,
	cD *gocudnn.ConvolutionD,
	fwdAlgo gocudnn.ConvFwdAlgo,
	bwdAlgodata gocudnn.ConvBwdDataAlgo,
	bwdAlgoFilt gocudnn.ConvBwdFiltAlgo,
	sizeinbytes gocudnn.SizeT) (*Layer, error) {
	datatype, _, _, err := wD.GetDescriptor()
	if err != nil {
		return nil, err
	}

	alpha := gocudnn.FindScalar(datatype, 1)
	alpha2 := gocudnn.FindScalar(datatype, 1)
	beta := gocudnn.FindScalar(datatype, 0)
	return &Layer{
		size:        sizeinbytes,
		cD:          cD,
		wD:          wD,
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
	return handle.ConvolutionForward(c.fwd.alpha, x.TensorD(), x.Mem(), c.wD, c.w, c.cD, c.fwdAlgo, wspace, c.fwd.beta, y.TensorD(), y.Mem())
}

//ForwardBiasActivation does the forward bias activation cudnn algorithm
func (c *Layer) ForwardBiasActivation(handle *gocudnn.Handle, x layers.IO, wpsace gocudnn.Memer, z layers.IO, aD *gocudnn.ActivationD, y layers.IO) error {
	return handle.ConvolutionBiasActivationForward(c.fwd.alpha, x.TensorD(), x.DMem(), c.wD, c.w, c.cD, c.fwdAlgo, wpsace, c.fwd.alpha2, z.TensorD(), z.DMem(), c.biasD, c.bias, aD, y.TensorD(), y.Mem())
}

//BackPropData performs the BackPropData
func (c *Layer) BackPropData(handle *gocudnn.Handle, wspace gocudnn.Memer, dx, dy *layers.IO) error {
	return handle.ConvolutionBackwardData(
		c.bwdd.alpha,
		c.wD,
		c.w,
		dy.TensorD(),
		dy.Mem(),
		c.cD,
		c.bwdAlgodata,
		wspace,
		c.bwdd.beta,
		dx.TensorD(),
		dx.DMem())

}

//BackPropFilter does the backward propagation for the filter You will pass a handle workspace memory x,dy layer.io
func (c *Layer) BackPropFilter(handle *gocudnn.Handle, wspace gocudnn.Memer, x, dy *layers.IO) error {
	return handle.ConvolutionBackwardFilter(c.bwdf.alpha, x.TensorD(), x.Mem(), dy.TensorD(), dy.DMem(), c.cD, c.bwdAlgoFilt, wspace, c.bwdf.beta, c.dwD, c.dw)

}

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
