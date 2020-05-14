package cnntranspose

import (
	"errors"
	"fmt"
	"sync"

	gocudnn "github.com/dereklstinson/gocudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/deconvolution"
	"github.com/dereklstinson/gocunets/layers"
)

const alphadefault = 1.0
const beta1default = 0.0
const beta2default = 1.0

//Layer is a struct that holds  filter, bias and convolution descriptors.
//The memory for w, dw, bias, dbias. The algos for forward, backward (data, filter) and the scalars for those algos. 1
type Layer struct {
	conv       *deconvolution.Ops
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
	//	train      trainer.Trainer
	//	btrain     trainer.Trainer
	pad      []int32
	dilation []int32
	stride   []int32
	//	l1b        float32
	//	l2b        float32
	//	l1w        float32
	//	l2w        float32
	mux sync.Mutex
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

/*
//UpdateWeights does the weight update
func (c *Layer) UpdateWeights(handle *cudnn.Handler, batch, epoch int) error {

	err := c.train.UpdateWeights(handle, c.dw, c.w, batch, epoch)

	if err != nil {
		return err
	}
	c.l1w, c.l2w = c.train.L1L2Loss()
	err = c.btrain.UpdateWeights(handle, c.dbias, c.bias, batch, epoch)
	if err != nil {
		return err
	}
	c.l1b, c.l2b = c.btrain.L1L2Loss()

	return nil
}

//L1L2Loss will return the L1 loss and L2 loss for the layer
func (c *Layer) L1L2Loss() (L1 float32, L2 float32) {

	return c.l1b + c.l1w, c.l2b + c.l2w
}

//LoadTrainer sets up the momentum trainer
func (c *Layer) LoadTrainer(handle *cudnn.Handler, forweights, forbias trainer.Trainer) error {
	var err error
	c.train = forweights
	err = trainer.CreateTrainingMem(handle, c.train, c.w)
	if err != nil {
		return err
	}
	c.btrain = forbias
	err = trainer.CreateTrainingMem(handle, c.btrain, c.bias)
	if err != nil {
		return err
	}
	return err
}
*/
//GetWeights gets the weights
func (c *Layer) GetWeights() []*layers.Tensor {
	return []*layers.Tensor{c.w, c.bias}
}

//GetDeltaWeights gets the delta weights
func (c *Layer) GetDeltaWeights() []*layers.Tensor {
	return []*layers.Tensor{c.dw, c.dbias}
}

//DeltaBias returns DeltaBias
func (c *Layer) DeltaBias() *layers.Tensor {
	return c.dbias
}

//Bias returns the Bias
func (c *Layer) Bias() *layers.Tensor {
	return c.bias
}

//DeltaWeights returns the deltaweights
func (c *Layer) DeltaWeights() *layers.Tensor {
	return c.dw
}

//Weights returns the weights
func (c *Layer) Weights() *layers.Tensor {
	return c.w
}

//OutputDimsFromInputTensor returns outputdims from an input tensor
func (c *Layer) OutputDimsFromInputTensor(input *layers.Tensor) ([]int32, error) {
	return c.conv.OutputDim(input.Volume, c.w.Volume)
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
	cpad, cstride, cdilation := make([]int32, len(pad)), make([]int32, len(stride)), make([]int32, len(dilation))
	copy(cpad, pad)
	copy(cstride, stride)
	copy(cdilation, dilation)
	conv, err := deconvolution.StageOperation(convmode, dtype, mtype, groupcount, cpad, cstride, cdilation)
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
		pad:      cpad,
		stride:   cstride,
		dilation: cdilation,
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

/*
//SetupEx creates a convolution layer
func SetupEx(handle *cudnn.Handler,
	frmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
	mtype gocudnn.MathType,
	groupcount int32,
	w, dw *layers.Tensor,
	b, db *layers.Tensor,
	cmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dialation []int32) (c *Layer, err error) {
	c = new(Layer)
	c.conv, err = convolution.StageOperation(cmode, dtype, mtype, groupcount, pad, stride, dialation)
	if err != nil {
		return nil, err
	}
	c.w, c.dw, c.bias, c.dbias = w, dw, b, db
	return c, nil
}
*/

//MakeRandom does what it says it will make the weights random considering the fanin
func (c *Layer) MakeRandom(h *cudnn.Handler, inputdims []int32) error {
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
	conv, err := deconvolution.StageOperation(convmode, dtype, mtype, groupcount, pad, stride, dialation)
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

/*
//SetAlphaScalars updates the alpha scalars in order of fwd, bwd-data,bwd-filter.
func (c *Layer) SetAlphaScalars(alphas []float64) error {
	if len(alphas) != 3 {
		return errors.New("alpha Scalar length needs to be 3")
	}

	c.fwd.alpha = alphas[0]
	c.bwdd.alpha = alphas[1]
	c.bwdf.alpha = alphas[2]
	return nil
}

//SetBetaScalars updates the alpha scalars in order of fwd, bwd-data,bwd-filter.
func (c *Layer) SetBetaScalars(betas []float64) error {
	if len(betas) != 3 {
		return errors.New("alpha Scalar length needs to be 3")
	}

	c.fwd.beta = betas[0]
	c.bwdd.beta = betas[1]
	c.bwdf.beta = betas[2]
	return nil
}
//NumAlphaScalars returns the number of alpha scalars which is used for fwd,bwd-data,bwd-filter.
func (c *Layer) NumAlphaScalars() int {
	return 3
}

//NumBetaScalars returns the number of beta scalars which is used for fwd,bwd-data,bwd-filter.
func (c *Layer) NumBetaScalars() int {
	return 3
}
*/

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
	out[1] = w[1]
	out[2] = findoutputdim(x[2], w[2], stride[0], padding[0], dilation[0])
	out[3] = findoutputdim(x[3], w[3], stride[1], padding[1], dilation[1])
	return out
}
func find4doutputdims4dNHWC(x, w, padding, stride, dilation []int32) []int32 {
	out := make([]int32, len(x))
	out[0] = x[0]
	out[1] = findoutputdim(x[1], w[1], stride[0], padding[0], dilation[0])
	out[2] = findoutputdim(x[2], w[2], stride[1], padding[1], dilation[1])
	out[3] = w[3]

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
	return (x-1)*s - 2*p + (((w - 1) * d) + 1)
}

//x=4, s=2, w=4, d=1, p=1 -> 6 - 2 + ((3 +1) -> 6 -2 +4 = 8
//x=4, s=2, w=4, d=3, p=4 -> 6 - 8 + 9 +1 = 8
//x=8, s=2, w=4, d=1, p=1 -> 14 - 2 + ((3 +1) -> 12 + 4= 16
//x=8, s=2, w=4, d=3, p=4 -> 14 - 8 + 10 = 16
func buildbias(handle *cudnn.Handler, weights *layers.Tensor) (*layers.Tensor, error) {
	frmt, dtype, dims, err := weights.Properties()
	if err != nil {
		return nil, err
	}
	outputmaps := dims[1]
	for i := 0; i < len(dims); i++ {

		dims[i] = int32(1)
	}
	dims[1] = outputmaps
	return layers.CreateTensor(handle, frmt, dtype, dims)
}
