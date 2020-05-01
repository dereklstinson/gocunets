package gocunets

/*


import (
	"github.com/dereklstinson/gocunets/layers/cnn"
)

//Convolution inherits *cnn.Layer
type Convolution struct {
	*cnn.Layer
}

//ConvolutionOptions are flags uesd in building a convolution layer
type ConvolutionOptions struct {
	Frmt  TensorFormat
	Dtype DataType
	Mtype MathType
	Cmode ConvolutionMode
}

//CreateBasicConvolution creates a convolution operation.
//Memory is placed into it
func CreateBasicConvolution(handle *Handle,
	frmt TensorFormat,
	dtype DataType,
	mtype MathType,
	groupcount int32,
	w, dw *Tensor,
	b, db *Tensor,
	cmode ConvolutionMode,
	pad,
	stride,
	dilation []int32) (c *Convolution, err error) {
	c = new(Convolution)
	c.Layer, err = cnn.SetupBasic(handle.Handler,
		frmt.TensorFormat,
		dtype.DataType,
		mtype.MathType,
		groupcount,
		w.Tensor, dw.Tensor,
		b.Tensor, db.Tensor,
		cmode.ConvolutionMode,
		pad, stride, dilation)
	return c, err
}

//CreateConvolution creates a convolution operation
func CreateConvolution(handle *Handle,
	frmt TensorFormat,
	dtype DataType,
	mtype MathType,
	groupcount int32,
	filterdims []int32,
	cmode ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	seed uint64) (c *Convolution, err error) {
	c = new(Convolution)
	c.Layer, err = cnn.Setup(handle.Handler,
		frmt.TensorFormat,
		dtype.DataType,
		mtype.MathType,
		groupcount,
		filterdims,
		cmode.ConvolutionMode,
		pad, stride, dilation, seed)
	return c, err
}
*/
