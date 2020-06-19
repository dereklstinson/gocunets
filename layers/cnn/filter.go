//Package cnn contains structs and methods used to do forward, and backward operations for convolution layers
package cnn

import (
	"errors"
	"fmt"

	gocudnn "github.com/dereklstinson/gocudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/convolution"
	"github.com/dereklstinson/gocunets/layers"
)

const alphadefault = 1.0
const beta1default = 0.0
const beta2default = 1.0

//Layer is a struct that holds  filter, bias and convolution descriptors.
//The memory for w, dw, bias, dbias. The algos for forward, backward (data, filter) and the scalars for those algos. 1
type Layer struct {
	conv       *convolution.Ops
	w          *layers.Tensor
	bias       *layers.Tensor
	dw         *layers.Tensor
	dbias      *layers.Tensor
	pindims    []int32
	wspacesize uint
	fwd        xtras
	bwdd       xtras
	bwdf       xtras
	datatype   gocudnn.DataType
	mathtype   gocudnn.MathType
	pad        []int32
	dilation   []int32
	stride     []int32
}

type xtras struct {
	alpha float64
	beta  float64
}

func appenderror(comment string, err error) error {
	return errors.New(comment + ": " + err.Error())
}

//ToggleDWeightsPrintValueForStringer toggles if DWeight values will be printed
func (c *Layer) ToggleDWeightsPrintValueForStringer() {
	c.dw.TogglePrintValueForStringer()
}

//ToggleWeightsPrintValueForStringer toggles if Weight values will be printed
func (c *Layer) ToggleWeightsPrintValueForStringer() {
	c.w.TogglePrintValueForStringer()
}

//ToggleDBiasPrintValueForStringer toggles if dBias values will be printed
func (c *Layer) ToggleDBiasPrintValueForStringer() {
	c.dbias.TogglePrintValueForStringer()
}

//ToggleBiasPrintValueForStringer toggles if Bias values will be printed
func (c *Layer) ToggleBiasPrintValueForStringer() {
	c.bias.TogglePrintValueForStringer()
}

func (c *Layer) String() string {
	return fmt.Sprintf("CnnTranspose Layer {\n%v\nWeights: %v\nBias: %v\nDWeights: %v\nDBias: %v\n}\n", c.conv, c.w, c.bias, c.dw, c.dbias)
}

//GetWeights gets the weights.  Order returned is weights, bias
func (c *Layer) GetWeights() []*layers.Tensor {
	return []*layers.Tensor{c.w, c.bias}
}

//GetDeltaWeights gets the delta weights.  Order returned is weights,bias
func (c *Layer) GetDeltaWeights() []*layers.Tensor {
	return []*layers.Tensor{c.dw, c.dbias}
}

//Bias returns the Bias
func (c *Layer) Bias() *layers.Tensor {
	return c.bias
}

//DeltaBias returns DeltaBias
func (c *Layer) DeltaBias() *layers.Tensor {
	return c.dbias
}

//DeltaWeights returns the deltaweights
func (c *Layer) DeltaWeights() *layers.Tensor {
	return c.dw
}

//Weights returns the weights
func (c *Layer) Weights() *layers.Tensor {
	return c.w
}

//OutputDims will return the dims for the output
func (c *Layer) OutputDims(inputdims []int32) []int32 {
	if len(inputdims) != 4 {
		return nil
	}

	frmt, _, dims, err := c.w.Properties()
	if err != nil {
		panic(err)
	}

	return find4doutputdims(inputdims, dims, c.conv.Pad(), c.conv.Stride(), c.conv.Dilation(), frmt)

}

//SetupBasic sets up a convolution layer with the memory for the gpu added to it.
//This can be used for layers that share the same memory, but might have different convolution properties.
func SetupBasic(handle *cudnn.Handler,
	frmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
	mtype gocudnn.MathType,
	groupcount int32,
	w, dw, b, db *layers.Tensor,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32) (*Layer, error) {

	conv, err := convolution.StageOperation(convmode, dtype, mtype, groupcount, pad, stride, dilation)
	if err != nil {
		fmt.Println("Error in Stage Operation")
		return nil, err
	}
	var fwd = xtras{
		alpha: 1,
		beta:  0,
	}
	var bwdd = xtras{
		alpha: 1,
		beta:  0,
	}
	var bwdf = xtras{
		alpha: 1,
		beta:  1,
	}
	return &Layer{
		mathtype: mtype,
		datatype: dtype,
		w:        w,
		dw:       dw,
		bias:     b,
		dbias:    db,
		conv:     conv,
		pad:      pad,
		stride:   stride,
		dilation: dilation,
		fwd:      fwd,
		bwdd:     bwdd,
		bwdf:     bwdf,
	}, nil
}

//Setup sets up the speed of the fwd and bwd algos dynamically.  guessinputdims is really for setting up the random weights.
func Setup(handle *cudnn.Handler,
	frmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
	mtype gocudnn.MathType,
	groupcount int32,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	seed uint64) (*Layer, error) {
	layer, err := layersetup(handle, frmt, dtype, mtype, groupcount, filterdims, convmode, pad, stride, dilation)
	if err != nil {
		fmt.Println("Error in layer setup")
		return nil, err
	}

	err = layer.bias.SetValues(handle, 0.00001)
	if err != nil {
		fmt.Println("Error in setvals")
		return nil, err
	}

	layer.pad, layer.stride, layer.dilation = pad, stride, dilation
	return layer, nil
}

//SetMathType sets the mathtype
func (c *Layer) SetMathType(mtype gocudnn.MathType) error {
	return c.conv.SetMathType(mtype)

}

//MakeRandom does what it says it will make the weights random considering the fanin
func (c *Layer) MakeRandom(h *cudnn.Handler, inputdims []int32) error {
	//	dims := c.w.T().Dims()
	fanin := int32(1)
	for i := 1; i < len(inputdims); i++ {
		fanin *= inputdims[i]
	}
	if h == nil {
		return c.w.SetRandom(0, 2.0, (float64)(fanin))

	}
	flg := c.w.Volume.DataType()
	if flg.Float() == c.w.Volume.DataType() {
		//	return h.GetCuRNG().NormalFloat32(c.w, c.w.SIB(), 0, 2*float32(math.Sqrt((2.0)))/float32(fanin))
	}
	return c.w.SetRandom(0, 2.0, (float64)(fanin))

}

//LayerSetup sets up the cnn layer to be built. But doesn't build it yet.
func layersetup(
	handle *cudnn.Handler,
	frmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
	mtype gocudnn.MathType,
	groupcount int32,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dialation []int32,
) (*Layer, error) {
	conv, err := convolution.StageOperation(convmode, dtype, mtype, groupcount, pad, stride, dialation)
	if err != nil {
		fmt.Println("Error in Stage Operation")
		return nil, err
	}
	w, err := layers.CreateTensor(handle, frmt, dtype, filterdims)
	if err != nil {
		fmt.Println("FilterDims ", filterdims)
		fmt.Println("error in building IO Weights")
		return nil, err
	}

	bias, err := buildbias(handle, w)
	if err != nil {
		fmt.Println("Error in building bias")
		return nil, err
	}
	dw, err := layers.CreateTensor(handle, frmt, dtype, filterdims)
	if err != nil {
		fmt.Println("FilterDims ", filterdims)
		fmt.Println("error in building IO Weights")
		return nil, err
	}

	dbias, err := buildbias(handle, w)
	if err != nil {
		fmt.Println("Error in building bias")
		return nil, err
	}
	return &Layer{
		conv:  conv,
		w:     w,
		bias:  bias,
		dw:    dw,
		dbias: dbias,
		fwd: xtras{
			alpha: alphadefault,
			//	alpha2: alphadefault,
			beta: beta1default,
		},
		bwdd: xtras{
			alpha: alphadefault,
			//	alpha2: alphadefault,
			beta: beta1default,
		},
		bwdf: xtras{
			alpha: alphadefault,
			//	alpha2: alphadefault,
			beta: beta2default,
		},
		datatype: dtype,
	}, nil
}

//SetForwardScalars sets the alpha and beta scalars, the defaults are alpha, 1, beta=0 and are initialized in the function FilterSetup
func (c *Layer) SetForwardScalars(alpha, beta float64) {
	c.fwd.alpha, c.fwd.beta = alpha, beta
}

//SetBackwardScalars sets the alpha and beta scalars, the defaults are alpha, 1, beta=0 and are initialized in the function FilterSetup
func (c *Layer) SetBackwardScalars(alpha, beta float64) {
	c.bwdd.alpha, c.bwdd.beta = alpha, beta
}

//SetOtherScalars sets the alpha and beta scalars for the weights, the defaults are alpha,  1, beta=1 and are initialized in the function FilterSetup
func (c *Layer) SetOtherScalars(alpha, beta float64) {
	c.bwdf.alpha, c.bwdf.beta = alpha, beta
}

func find4doutputdims(x, w, padding, stride, dilation []int32, frmt gocudnn.TensorFormat) []int32 {
	var flag gocudnn.TensorFormat
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
func buildbias(handle *cudnn.Handler, weights *layers.Tensor) (*layers.Tensor, error) {
	frmt, dtype, dims, err := weights.Properties()
	if err != nil {
		return nil, err
	}
	outputmaps := dims[0]
	for i := 0; i < len(dims); i++ {

		dims[i] = int32(1)
	}
	dims[1] = outputmaps
	return layers.CreateTensor(handle, frmt, dtype, dims)
}
