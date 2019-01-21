package cnn

import (
	"fmt"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//This stuff is experimental and it is just easier to set it up this way than to make a whole new package.  This is only for the convtranspose layer.

//SetupDynamicReverse sets up the speed of the fwd and bwd algos dynamically.  guessinputdims is really for setting up the random weights.
func SetupDynamicReverse(handle *cudnn.Handler,
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
	//	guessinputdims []int32,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	managedmem bool) (*Layer, error) {
	layer, err := layersetupreverse(handle, frmt, dtype, filterdims, convmode, pad, stride, dilation)
	if err != nil {
		return nil, err
	}
	err = layer.MakeRandomFromFaninDims(handle, filterdims)
	if err != nil {
		return nil, err
	}

	err = layer.bias.T().SetValues(handle, 0.0001)
	if err != nil {
		return nil, err
	}

	layer.pad, layer.stride, layer.dilation = pad, stride, dilation
	return layer, nil
}

//ReverseForwardProp performs the ForwardProp
func (c *Layer) ReverseForwardProp(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {

	err := c.conv.BwdPropData(
		handle,
		c.fwd.alpha,
		c.w.T(),
		x.T(), //y.DeltaT(),
		wspace,
		c.fwd.beta,
		y.T(), //x.DeltaT(),
	)
	if err != nil {
		return utils.ErrorWrapper("BackPropData(reverse so its forwardprop): ", err)
	}

	err = utils.ErrorWrapper("AddBias", y.T().AddTo(handle, c.bias.T(), 1.0, 1.0))
	if err != nil {
		_, _, wdims, _ := c.w.Properties()
		fmt.Println("xdims", x.T().TD().Dims(), "ydims", x.DeltaT().TD().Dims(), "bias dims", c.bias.T().TD().Dims(), "wdims", wdims)
		return err
	}

	return nil
}

//ReverseBackPropFilterData does the backprop for the data and the filter
func (c *Layer) ReverseBackPropFilterData(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	var err error
	if x.IsInput() == true {
		return c.ReverseBackPropFilter(handle, wspace, x, y)
	}
	err = utils.ErrorWrapper("ReverseBackPropData: ", c.ReverseBackPropData(handle, wspace, x, y))

	if err != nil {
		return err
	}
	err = utils.ErrorWrapper("ReverseBackPropFilter: ", c.ReverseBackPropFilter(handle, wspace, x, y))

	if err != nil {
		return err
	}
	return nil
}

//ReverseBackPropData performs the BackPropData
func (c *Layer) ReverseBackPropData(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	if x.IsInput() == true {
		return nil
	}
	err := c.conv.FwdProp(handle,
		c.bwdd.alpha,
		y.DeltaT(), //x.T(),
		c.w.T(),
		wspace,
		c.bwdd.beta,
		x.DeltaT(), //y.T(),
	)
	if err != nil {
		return err
	}

	return nil

}

//ReverseBackPropFilter does the backward propagation for the filter You will pass a handle workspace memory x,dy layer.io
func (c *Layer) ReverseBackPropFilter(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	err := c.conv.BwdPropFilt(
		handle,
		c.bwdf.alpha,
		y.DeltaT(), //x.T(),     ///This might need switched
		x.T(),      // y.DeltaT(), //This might need switched
		wspace,
		c.bwdf.beta,
		c.w.DeltaT())
	if err != nil {
		return utils.ErrorWrapper("Filter", err)
	}

	err = c.conv.BwdBias(
		handle,
		c.bwdf.alpha,
		y.DeltaT(), //	y.DeltaT(),
		c.bwdf.beta,
		c.bias.DeltaT())
	if err != nil {
		return utils.ErrorWrapper("Bias", err)
	}
	return nil

}

//MakeReverseOutputTensor makes the output tensor of the reverse convolution layer
func (c *Layer) MakeReverseOutputTensor(handle *cudnn.Handler, input *layers.IO) (*layers.IO, error) {
	xdims := input.T().TD().Dims()
	//cvol := float32(utils.FindVolumeInt32(xdims, nil))
	//ratio := float32(input.T().MaxVol()) / cvol
	frmt, dtype, wdims, err := c.w.Properties()
	if err != nil {
		return nil, err
	}
	dims := findreverse4doutputdims(xdims, wdims, c.pad, c.stride, c.dilation, frmt)
	if err != nil {
		return nil, err
	}

	output, err := layers.BuildIO(handle, frmt, dtype, dims)
	if err != nil {
		return nil, err
	}
	return output, nil
}

func findreverse4doutputdims(x, w, padding, stride, dilation []int32, frmt cudnn.TensorFormat) []int32 {
	var flag cudnn.TensorFormatFlag
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
func findreverseoutputdim(x, w, s, p, d int32) int32 {
	// output = 1+ (input + (2*padding) - (((filter-1)*dilation)+1))/slide
	//	(input-1)*slide = (output +2*padding)-(((filter-1)*dilation)+1)
	//output= 2*padding-(((filter-1)*dilation)+1)-(input-1)*slide
	// input = 1 + (output + (2*padding) - (((filter-1)*dilation)+1))/slide
	//  slide *(input-1) = output + (2*padding) - (((filter-1)*dilation)+1)
	//  output = (slide *(input-1)) - (2*padding) + (((filter-1)*dilation)+1)
	return (s * (x - 1)) - (2 * p) + (((w - 1) * d) + 1)
}

//LayerSetup sets up the cnn layer to be built. But doesn't build it yet.
func layersetupreverse(
	handle *cudnn.Handler,
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dialation []int32,

) (*Layer, error) {
	conv, err := convolution.StageOperation(convmode, dtype, pad, stride, dialation)
	if err != nil {
		return nil, err
	}
	w, err := layers.BuildIO(handle, frmt, dtype, filterdims)
	if err != nil {
		return nil, err
	}
	/*
		sizeinbytes, err := w.T().Size()
		if err != nil {
			return nil, err
		}
	*/
	bias, err := buildbiasreverse(handle, w)
	if err != nil {
		return nil, err
	}

	alpha := 1.0
	alpha2 := 1.0
	beta := 0.0
	beta2 := 0.0
	return &Layer{
		//size: sizeinbytes,
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
		datatype: dtype,
	}, nil
}
func buildbiasreverse(handle *cudnn.Handler, weights *layers.IO) (*layers.IO, error) {
	frmt, dtype, dims, err := weights.Properties()
	if err != nil {
		return nil, err
	}
	outputmaps := dims[1]
	for i := 0; i < len(dims); i++ {

		dims[i] = int32(1)
	}
	dims[1] = outputmaps
	return layers.BuildIO(handle, frmt, dtype, dims)
}
