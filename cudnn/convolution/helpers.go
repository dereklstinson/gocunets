package convolution

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

/*
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
*/

/*
ForwardPerformance is wrapper for gocudnn.ConvFwdAlgoPerformance with its exported Values

type ConvFwdAlgoPerformance struct{
	Algo        ConvFwdAlgo --Algo is the flag for the cudnn algorithm
	Status      Status    -- error occurance while running test
	Time        float32  -- time it takes to do the algo
	Memory      SizeT  --size of workspace memory to be passed
	Determinism Determinism  --flag
	MathType    MathType -- flag
}
*/
type ForwardPerformance gocudnn.ConvFwdAlgoPerformance

//Print will print the forward performance to console
func (f ForwardPerformance) Print() {
	f.Print()
}

/*
BackDataPerformance is a wrapper for gocudnn.ConvBwdDataAlgoPerformance

type ConvBwdDataAlgoPerformance struct {
	Algo        ConvFwdAlgo --flag --- Algo is the flag for the cudnn algorithm
	Status      Status    -- error occurance while running test
	Time        float32  -- time it takes to do the algo
	Memory      SizeT  --size of workspace memory to be passed
	Determinism Determinism  --flag determins if the algo is deterministic
	MathType    MathType -- flag chooses the mathtype - tensorcores?
}
*/
type BackDataPerformance gocudnn.ConvBwdDataAlgoPerformance

//Print will print forward performance to console
func (b BackDataPerformance) Print() {
	b.Print()
}

/*
BackFilterPerformance is a wrapper for gocudnn.ConvBwdFiltAlgoPerformance
type ConvBwdFiltAlgoPerformance struct {
	Algo        ConvFwdAlgo --flag --- Algo is the flag for the cudnn algorithm
	Status      Status    -- error occurance while running test
	Time        float32  -- time it takes to do the algo
	Memory      SizeT  --size of workspace memory to be passed
	Determinism Determinism  --flag determins if the algo is deterministic
	MathType    MathType -- flag chooses the mathtype - tensorcores?
}
*/
type BackFilterPerformance gocudnn.ConvBwdFiltAlgoPerformance

//SetFwdPerformanceAlgo will sets the convolution algorithm
func (c *Ops) SetFwdPerformanceAlgo(fwd ForwardPerformance) {
	c.perfforward = fwd
	c.fwddesc.SetMathType(fwd.MathType)
}

//SetBwdDataPerformanceAlgo will sets the convolution algorithm
func (c *Ops) SetBwdDataPerformanceAlgo(bwddata BackDataPerformance) {
	c.perfbackdata = bwddata
	c.bwdddesc.SetMathType(bwddata.MathType)
}

//SetBwdFiltPerformanceAlgo will sets the convolution algorithm
func (c *Ops) SetBwdFiltPerformanceAlgo(bwdfilt BackFilterPerformance) {
	c.perfbackfilt = bwdfilt
	c.bwdfdesc.SetMathType(bwdfilt.MathType)
}

//GetFwdAlgoPerfList gets a list of forward performance stats
func (c *Ops) GetFwdAlgoPerfList(handle *cudnn.Handler, x, w, y *tensor.Volume) ([]ForwardPerformance, error) {
	maxfwd, err := c.helper.Funcs.Fwd.GetConvolutionForwardAlgorithmMaxCount(handle.Cudnn())
	if err != nil {
		return nil, err
	}
	fwdlist, err := c.helper.Funcs.Fwd.FindConvolutionForwardAlgorithm(handle.Cudnn(), x.TD(), w.FD(), c.fwddesc, y.TD(), maxfwd)
	if err != nil {
		return nil, err
	}
	fwper := make([]ForwardPerformance, 0)
	for i := range fwdlist {

		err = fwdlist[i].Status.Error("Not Available")
		if err != nil {

		} else {
			fwper = append(fwper, ForwardPerformance(fwdlist[i]))
		}

	}
	return fwper, nil
}

//GetBwdDataAlgoPerfList gets a list of backward data performance stats to set the convolution algo
func (c *Ops) GetBwdDataAlgoPerfList(handle *cudnn.Handler, dx, w, dy *tensor.Volume) ([]BackDataPerformance, error) {
	maxbwddata, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataAlgorithmMaxCount(handle.Cudnn())
	if err != nil {
		return nil, err
	}
	bwddata, err := c.helper.Funcs.Bwd.FindConvolutionBackwardDataAlgorithm(handle.Cudnn(), w.FD(), dy.TD(), c.bwdddesc, dx.TD(), maxbwddata)
	if err != nil {
		return nil, err
	}
	bwper := make([]BackDataPerformance, 0)
	for i := range bwddata {

		err = bwddata[i].Status.Error("Not Available")
		if err != nil {

		} else {
			bwper = append(bwper, BackDataPerformance(bwddata[i]))
		}

	}
	return bwper, nil
}

//GetBwdFiltAlgoPerfList gets a list of backward filter stats
func (c *Ops) GetBwdFiltAlgoPerfList(handle *cudnn.Handler, x, dw, dy *tensor.Volume) ([]BackFilterPerformance, error) {
	maxbwdfilt, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithmMaxCount(handle.Cudnn())
	if err != nil {
		return nil, err
	}
	bwdfilt, err := c.helper.Funcs.Bwd.FindConvolutionBackwardFilterAlgorithm(handle.Cudnn(), x.TD(), dy.TD(), c.bwdfdesc, dw.FD(), maxbwdfilt)
	if bwdfilt != nil {
		return nil, err
	}
	bwfper := make([]BackFilterPerformance, 0)
	for i := range bwdfilt {

		err = bwdfilt[i].Status.Error("Not Available")
		if err != nil {

		} else {
			bwfper = append(bwfper, BackFilterPerformance(bwdfilt[i]))
		}

	}
	return bwfper, nil
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
		return c.fwddesc.GetConvolution2dForwardOutputDim(input.TD(), filter.FD())
	}
	return c.fwddesc.GetConvolutionNdForwardOutputDim(input.TD(), filter.FD())

}
