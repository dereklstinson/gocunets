package cnn

import (
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is a struct that holds a cnn layer info
type Layer struct {
	cD          *gocudnn.ConvolutionD
	wD          *gocudnn.FilterD
	dwD         *gocudnn.FilterD
	w           gocudnn.Memer
	dw          gocudnn.Memer
	size        gocudnn.SizeT
	fwdAlgo     gocudnn.ConvFwdAlgo
	bwdAlgodata gocudnn.ConvBwdDataAlgo
	bwdAlgoFilt gocudnn.ConvBwdFiltAlgo
	fwd         xtras
	bwdd        xtras
	bwdf        xtras
}

type xtras struct {
	wspace gocudnn.Memer
	alpha  gocudnn.CScalar
	beta   gocudnn.CScalar
}

//FilterSetup sets up the cnn layer to be built. But doesn't build it yet.
func FilterSetup(input *gocudnn.TensorD,
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
	beta := gocudnn.FindScalar(datatype, 0)
	return &Layer{
		size:        sizeinbytes,
		cD:          cD,
		wD:          wD,
		fwdAlgo:     fwdAlgo,
		bwdAlgodata: bwdAlgodata,
		bwdAlgoFilt: bwdAlgoFilt,
		fwd: xtras{

			alpha: alpha,
			beta:  beta,
		},
		bwdd: xtras{

			alpha: alpha,
			beta:  beta,
		},
		bwdf: xtras{

			alpha: alpha,
			beta:  beta,
		},
	}, nil
}

//SetFwdScalars sets the alpha and beta scalars, the defaults are alpha=1, beta=0 and are initialized in the function FilterSetup
func (c *Layer) SetFwdScalars(alpha gocudnn.CScalar, beta gocudnn.CScalar) {
	c.fwd.alpha, c.fwd.beta = alpha, beta
}

//SetFwdWorkSpace sets the fwd workspace
func (c *Layer) SetFwdWorkSpace(bytesize gocudnn.SizeT) error {
	var err error
	c.fwd.wspace, err = gocudnn.Malloc(bytesize)
	return err
}

//SetBwdDataWorkSpace sets the backward Data workspace
func (c *Layer) SetBwdDataWorkSpace(bytesize gocudnn.SizeT) error {
	var err error
	c.fwd.wspace, err = gocudnn.Malloc(bytesize)
	return err
}

//SetBwdFilterWorkSpace sets the backward filter workspace
func (c *Layer) SetBwdFilterWorkSpace(bytesize gocudnn.SizeT) error {
	var err error
	c.fwd.wspace, err = gocudnn.Malloc(bytesize)
	return err
}

//SetBwdDataScalars sets the alpha and beta scalars, the defaults are alpha=1, beta=0 and are initialized in the function FilterSetup
func (c *Layer) SetBwdDataScalars(alpha gocudnn.CScalar, beta gocudnn.CScalar) {
	c.bwdd.alpha, c.bwdd.beta = alpha, beta
}

//SetBwdFilterScalars sets the alpha and beta scalars, the defaults are alpha=1, beta=0 and are initialized in the function FilterSetup
func (c *Layer) SetBwdFilterScalars(alpha gocudnn.CScalar, beta gocudnn.CScalar) {
	c.bwdf.alpha, c.bwdf.beta = alpha, beta
}

//ForwardProp performs the ForwardProp
func (c *Layer) ForwardProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	return handle.ConvolutionForward(c.fwd.alpha, x.TensorD(), x.Mem(), c.wD, c.w, c.cD, c.fwdAlgo, c.fwd.wspace, c.fwd.beta, y.TensorD(), y.Mem())
}

//BackPropData performs the BackPropData
func (c *Layer) BackPropData(handle *gocudnn.Handle, dx, dy *layers.IO) error {
	return handle.ConvolutionBackwardData(
		c.bwdd.alpha,
		c.wD,
		c.w,
		dy.TensorD(),
		dy.Mem(),
		c.cD,
		c.bwdAlgodata,
		c.bwdd.wspace,
		c.bwdd.beta,
		dx.TensorD(),
		dx.DMem())

}

//BackPropFilter does the backward propagation for the filter
func (c *Layer) BackPropFilter(handle *gocudnn.Handle, x, dy *layers.IO) error {
	return handle.ConvolutionBackwardFilter(c.bwdf.alpha, x.TensorD(), x.Mem(), dy.TensorD(), dy.DMem(), c.cD, c.bwdAlgoFilt, c.bwdf.wspace, c.bwdf.beta, c.dwD, c.dw)

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
