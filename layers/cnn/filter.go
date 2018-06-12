package cnn

import (
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is a struct that holds a cnn layer info
type Layer struct {
	cD          *gocudnn.ConvolutionD
	wD          *gocudnn.FilterD
	w           gocudnn.Memer
	dw          gocudnn.Memer
	size        gocudnn.SizeT
	fwdalgo     gocudnn.ConvFwdAlgo
	bwdfiltalgo gocudnn.ConvBwdFiltAlgo
	bwddataalgo gocudnn.ConvBwdDataAlgo
	wspace      gocudnn.Memer
	alpha       gocudnn.CScalar
	beta        gocudnn.CScalar
}
type memory struct {
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
		size:  sizeinbytes,
		cD:    cD,
		wD:    wD,
		alpha: alpha,
		beta:  beta,
	}, nil
}

//SetScalars sets the alpha and beta scalars, the defaults are alpha=1, beta=0 and are initialized in the function FilterSetup
func (c *Layer) SetScalars(alpha gocudnn.CScalar, beta gocudnn.CScalar) {
	c.alpha, c.beta = alpha, beta
}

//ForwardProp performs the ForwardProp
func (c *Layer) ForwardProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	return handle.ConvolutionForward(c.alpha, x.TensorD(), x.Mem(), c.wD, c.w, c.cD, c.fwdalgo, c.wspace, c.beta, y.TensorD(), y.Mem())
}

//BackPropData performs the BackPropData
func (c *Layer) BackPropData(handle *gocudnn.Handle, dx, dy *layers.IO) {
	handle.ConvolutionBackwardData(
		c.alpha,
		c.wD,
		c.w,
		dy.TensorD(),
		dy.Mem(),
		c.cD,
		c.bwddataalgo,
		c.wspace,
		c.beta,
		dx.TensorD(),
		dx.DMem())

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
