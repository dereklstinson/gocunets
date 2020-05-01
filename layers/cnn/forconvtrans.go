package cnn

/*
import (
	"errors"
	"fmt"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/convolution"
	"github.com/dereklstinson/gocunets/layers"
	"github.com/dereklstinson/gocunets/utils"
	gocudnn "github.com/dereklstinson/gocudnn"
)

//This stuff is experimental and it is just easier to set it up this way than to make a whole new package.  This is only for the convtranspose layer.

//SetupReverse sets up the speed of the fwd and bwd algos dynamically.  guessinputdims is really for setting up the random weights.
func SetupReverse(handle *cudnn.Handler,
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
	layer, err := layersetupreverse(handle, frmt, dtype, mtype, groupcount, filterdims, convmode, pad, stride, dilation)
	if err != nil {
		return nil, err
	}

	err = layer.bias.SetValues(handle, 0.0001)
	if err != nil {
		return nil, err
	}

	layer.pad, layer.stride, layer.dilation = pad, stride, dilation
	return layer, nil
}

//ReverseForwardProp performs the ForwardProp
func (c *Layer) ReverseForwardProp(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.Tensor) error {

	err := c.conv.BackwardData(
		handle,
		c.fwd.alpha,
		c.w.Volume,
		x.Volume, //y.DeltaT(),
		wspace,
		c.fwd.beta,
		y.Volume, //x.DeltaT(),
	)
	if err != nil {
		//	panic(utils.ErrorWrapper("BackPropData(reverse so its forwardprop): ", err))
		return utils.ErrorWrapper("BackPropData(reverse so its forwardprop): ", err)
	}

	err = utils.ErrorWrapper("AddBias", y.AddTo(handle, c.bias.Volume, 1.0, 1.0))
	if err != nil {
		_, _, wdims, _ := c.w.Properties()
		fmt.Println("xdims", x.Dims(), "ydims", y.Dims(), "bias dims", c.bias.Dims(), "wdims", wdims)
		return err
	}

	return nil
}

//ReverseBackPropFilterData does the backprop for the data and the filter
func (c *Layer) ReverseBackPropFilterData(handle *cudnn.Handler, wspacedata, wspacefilter *nvidia.Malloced, x, dx, dy *layers.Tensor) error {
	var err error
	if dx == nil {
		return c.ReverseBackPropFilter(handle, wspacefilter, x, dy)
	}
	err = utils.ErrorWrapper("ReverseBackPropFilter: ", c.ReverseBackPropFilter(handle, wspacefilter, x, dy))

	if err != nil {
		return err
	}
	err = utils.ErrorWrapper("ReverseBackPropData: ", c.ReverseBackPropData(handle, wspacedata, dx, dy))

	return nil
}

//ReverseBackPropData performs the BackPropData
func (c *Layer) ReverseBackPropData(handle *cudnn.Handler, wspace *nvidia.Malloced, dx, dy *layers.Tensor) error {
	if dx == nil {
		return nil
	}
	err := c.conv.Forward(handle,
		c.bwdd.alpha,
		dy.Volume, //x.T(),
		c.w.Volume,
		wspace,
		c.bwdd.beta,
		dx.Volume, //y.T(),
	)
	if err != nil {
		return err
	}

	return nil

}

//ReverseBackPropFilter does the backward propagation for the filter You will pass a handle workspace memory x,dy layer.io
func (c *Layer) ReverseBackPropFilter(handle *cudnn.Handler, wspace *nvidia.Malloced, x, dy *layers.Tensor) error {
	err := c.conv.BackwardFilter(
		handle,
		c.bwdf.alpha,
		dy.Volume, // y.DeltaT(), //This might need switched
		x.Volume,  //x.T(),     ///This might need switched
		wspace,
		c.bwdf.beta,
		c.dw.Volume)
	if err != nil {
		//	fmt.Printf("\n ReverseConv: %v,", c.conv)
		//	fmt.Printf("\n\ndy: %v\nw: %v\nx: %v\n", dy, c.DeltaWeights(), x)
		//	panic(utils.ErrorWrapper("Filter", err))
		return utils.ErrorWrapper("Filter", err)
	}

	err = c.conv.BackwardBias(
		handle,
		c.bwdf.alpha,
		dy.Volume, //	y.DeltaT(),
		c.bwdf.beta,
		c.dbias.Volume)
	if err != nil {
		return utils.ErrorWrapper("Bias", err)
	}
	return nil

}

//FindReverseOutputDims returns the outputdims considering the input recieved
func (c *Layer) FindReverseOutputDims(input *layers.Tensor) ([]int32, error) {
	xdims := input.Dims()
	//cvol := float32(utils.FindVolumeInt32(xdims, nil))
	//ratio := float32(input.T().MaxVol()) / cvol
	frmt, _, wdims, err := c.w.Properties()
	if err != nil {
		return nil, err
	}
	return findreverse4doutputdims(xdims, wdims, c.pad, c.stride, c.dilation, frmt), nil
}

//MakeReverseOutputTensor makes the output tensor of the reverse convolution layer
func (c *Layer) MakeReverseOutputTensor(handle *cudnn.Handler, input *layers.Tensor) (*layers.Tensor, error) {
	xdims := input.Dims()
	//cvol := float32(utils.FindVolumeInt32(xdims, nil))
	//ratio := float32(input.T().MaxVol()) / cvol
	err := c.MakeRandom(xdims)
	if err != nil {
		return nil, err
	}
	frmt, dtype, wdims, err := c.w.Properties()
	if err != nil {
		return nil, err
	}
	dims := findreverse4doutputdims(xdims, wdims, c.pad, c.stride, c.dilation, frmt)
	if err != nil {
		return nil, err
	}

	output, err := layers.CreateTensor(handle, frmt, dtype, dims)
	if err != nil {
		return nil, err
	}
	return output, nil
}


////MakeReverseOutputTensorInference makes the output tensor of the reverse convolution layer inference mode so only contains x
//func (c *Layer) MakeReverseOutputTensorInference(handle *cudnn.Handler, input *layers.Tensor) (*layers.Tensor, error) {
//	xdims := input.Dims()
//
//	frmt, dtype, wdims, err := c.w.Properties()
//	if err != nil {
//		return nil, err
//	}
//	dims := findreverse4doutputdims(xdims, wdims, c.pad, c.stride, c.dilation, frmt)
//	if err != nil {
//		return nil, err
//	}
//
//	output, err := layers.CreateTensor(handle, frmt, dtype, dims)
//	if err != nil {
//		return nil, err
//	}
//	return output, nil
//}

func findreverse4doutputdims(x, w, padding, stride, dilation []int32, frmt gocudnn.TensorFormat) []int32 {
	var flag gocudnn.TensorFormat
	if frmt == flag.NCHW() {
		return findreverse4doutputdims4dNCHW(x, w, padding, stride, dilation)
	}
	return findreverse4doutputdims4dNHWC(x, w, padding, stride, dilation)
}
func findreverse4doutputdims4dNCHW(x, w, padding, stride, dilation []int32) []int32 {
	out := make([]int32, len(x))
	out[0] = x[0]
	out[1] = w[1]
	out[2] = findreverseoutputdim(x[2], w[2], stride[0], padding[0], dilation[0])
	out[3] = findreverseoutputdim(x[3], w[3], stride[1], padding[1], dilation[1])
	return out
}
func findreverse4doutputdims4dNHWC(x, w, padding, stride, dilation []int32) []int32 {
	out := make([]int32, len(x))
	out[0] = x[0]
	out[1] = findreverseoutputdim(x[1], w[1], stride[0], padding[0], dilation[0])
	out[2] = findreverseoutputdim(x[2], w[2], stride[1], padding[1], dilation[1])
	out[3] = w[1]

	return out
}
func findreverseoutputdim(y, w, s, p, d int32) int32 {
	// output = 1+ (input + (2*padding) - (((filter-1)*dilation)+1))/slide
	// y-1 = (x+(2*p)) - (((w-1)*d)+1))/slide
	// ((y-1)*s)=x+(2*p)-(((w-1)*d)+1)
	// ((y-1)*s)-(2*p)+(((w-1)*d)+1)=x
	// x=((y-1)*s)-(2*p)+(((w-1)*d)+1)

	return (s * (y - 1)) - (2 * p) + (((w - 1) * d) + 1)
}

//LayerSetup sets up the cnn layer to be built. But doesn't build it yet.
func layersetupreverse(
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
		return nil, err
	}
	w, err := layers.CreateTensor(handle, frmt, dtype, filterdims)
	if err != nil {
		return nil, err
	}
	dw, err := layers.CreateTensor(handle, frmt, dtype, filterdims)
	if err != nil {
		return nil, err
	}

	//	sizeinbytes, err := w.T().Size()
	//	if err != nil {
	//		return nil, err
	//	}
	//
	bias, err := buildbiasreverse(handle, w)
	if err != nil {
		return nil, err
	}
	dbias, err := buildbiasreverse(handle, w)
	if err != nil {
		return nil, err
	}

	alpha := 1.0
	//	alpha2 := 1.0
	beta := 0.0
	beta2 := 0.0
	return &Layer{
		//size: sizeinbytes,
		conv: conv,

		w:     w,
		bias:  bias,
		dw:    dw,
		dbias: dbias,
		fwd: xtras{
			alpha: alpha,
			//		alpha2: alpha2,
			beta: beta,
		},
		bwdd: xtras{
			alpha: alpha,
			//		alpha2: alpha2,
			beta: beta,
		},
		bwdf: xtras{
			alpha: alpha,
			//	alpha2: alpha2,
			beta: beta2,
		},
		datatype: dtype,
	}, nil
}
func buildbiasreverse(handle *cudnn.Handler, weights *layers.Tensor) (*layers.Tensor, error) {
	frmt, dtype, dims, err := weights.Properties()
	if err != nil {
		return nil, err
	}
	flag := frmt

	bdims := make([]int32, len(dims))
	for i := 0; i < len(bdims); i++ {
		bdims[i] = int32(1)
	}
	switch frmt {
	case flag.NCHW():

		bdims[1] = dims[1]

	case flag.NHWC():

		bdims[len(dims)-1] = dims[len(dims)-1]
	default:
		return nil, errors.New("buildreversebias: Unsupported tensor format")
	}

	return layers.CreateTensor(handle, frmt, dtype, bdims)
}
*/
