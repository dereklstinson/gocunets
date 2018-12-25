package convolution

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

func FindConvolution2DParams(x, w, y *tensor.Volume) (pad, stride, dilation []int32) {
	var (
		maxpadh      int32
		maxpadw      int32
		maxstrideh   int32
		maxstridew   int32
		maxdilationh int32
		maxdilationw int32
	)
	switch w.Format() {
	case cudnn.TensorFormatFlag{}.NHWC():
		wdims := w.Dims()
		maxstrideh = wdims[1]
		maxstridew = wdims[2]
		maxpadh = wdims[1] - 1
		maxpadw = wdims[2] - 1
	default:
		wdims := w.Dims()
		maxstrideh = wdims[2]
		maxstridew = wdims[3]
		maxpadh = wdims[2] - 1
		maxpadw = wdims[3] - 1

	}
	fmt.Println(maxpadh, maxpadw, maxstrideh, maxstridew, maxdilationh, maxdilationw)
	w.Dims()
	return
}

func findoutputdim(x, w, s, p, d int32) int32 {
	return 1 + (x+2*p-(((w-1)*d)+1))/s
}
func findpad(x, w, s, d, y int32) int32 {

	return (((y - 1) * s) - x + (((w - 1) * d) + 1)) / 2
}
func findslide(x, w, p, d, y int32) int32 {
	return (x + 2*p - (((w - 1) * d) + 1)) / (y - 1)
}
func finddilation(x, w, p, s, y int32) int32 {
	return -((s * (y - 1)) - x - (2 * p) + 1) / (w - 1)

}
func findpadandstrideanddilation(x, y, w int32) (s, p, d int32) {
	//	output = 1+ (input + (2*padding) - (((filter-1)*dilation)+1))/slide

	//first lets asume only slide of one and dilation of one we will see if the it fits inside the padding
	minwithpad := findoutputdim(x, w, 1, 0, 1)
	maxwithpad := findoutputdim(x, w, 1, w-1, 1)
	if y >= minwithpad && y <= maxwithpad {
		findpad()
	}
	return
}

//AlgoLists Algo lists returns slices of performances for the fwd algos and bwd algos
func (c *Ops) AlgoLists(handle *cudnn.Handler, x, dx, w, dw, y, dy *tensor.Volume) ([]gocudnn.ConvFwdAlgoPerformance, []gocudnn.ConvBwdDataAlgoPerformance, []gocudnn.ConvBwdFiltAlgoPerformance, error) {
	maxfwd, err := c.helper.Funcs.Fwd.GetConvolutionForwardAlgorithmMaxCount(handle.Cudnn())
	if err != nil {
		return nil, nil, nil, err
	}
	fwdlist, err := c.helper.Funcs.Fwd.FindConvolutionForwardAlgorithm(handle.Cudnn(), x.TD(), w.FD(), c.desc, y.TD(), maxfwd)
	if err != nil {
		return nil, nil, nil, err
	}
	maxbwddata, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataAlgorithmMaxCount(handle.Cudnn())
	if err != nil {
		return nil, nil, nil, err
	}
	bwddata, err := c.helper.Funcs.Bwd.FindConvolutionBackwardDataAlgorithm(handle.Cudnn(), w.FD(), dy.TD(), c.desc, dx.TD(), maxbwddata)
	if err != nil {
		return nil, nil, nil, err
	}
	maxbwdfilt, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithmMaxCount(handle.Cudnn())
	if err != nil {
		return nil, nil, nil, err
	}
	bwdfilt, err := c.helper.Funcs.Bwd.FindConvolutionBackwardFilterAlgorithm(handle.Cudnn(), x.TD(), dy.TD(), c.desc, dw.FD(), maxbwdfilt)
	if bwdfilt != nil {
		return nil, nil, nil, err
	}
	return fwdlist, bwddata, bwdfilt, nil
}

//OutputDim will return the dims of what the output tensor should be
func (c *Ops) OutputDim(input *tensor.Volume, filter *tensor.Volume) ([]int32, error) {
	_, _, dims, err := input.Properties()
	if err != nil {
		return nil, err
	}
	_, _, fdims, err := filter.Properties()

	if err != nil {
		return nil, err
	}

	if len(dims) != len(fdims) {
		return nil, errors.New("length of dims not same")
	}
	if len(dims) == 4 {
		return c.desc.GetConvolution2dForwardOutputDim(input.TD(), filter.FD())
	}
	return c.desc.GetConvolutionNdForwardOutputDim(input.TD(), filter.FD())

}
