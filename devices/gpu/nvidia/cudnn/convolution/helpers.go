package convolution

import (
	"fmt"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
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
func tofwrdperf(x gocudnn.ConvFwdAlgoPerformance) ForwardPerformance {
	return ForwardPerformance{
		Algo:        x.Algo,
		Status:      x.Status,
		Time:        x.Time,
		Memory:      x.Memory,
		Determinism: x.Determinism,
		MathType:    x.MathType,
	}
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

func tobwddperf(x gocudnn.ConvBwdDataAlgoPerformance) BackDataPerformance {
	return BackDataPerformance{
		Algo:        x.Algo,
		Status:      x.Status,
		Time:        x.Time,
		Memory:      x.Memory,
		Determinism: x.Determinism,
		MathType:    x.MathType,
	}
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

func tobwdfperf(x gocudnn.ConvBwdFiltAlgoPerformance) BackFilterPerformance {
	return BackFilterPerformance{
		Algo:        x.Algo,
		Status:      x.Status,
		Time:        x.Time,
		Memory:      x.Memory,
		Determinism: x.Determinism,
		MathType:    x.MathType,
	}
}

//SetFwdPerformanceAlgo will sets the convolution algorithm
func (c *Ops) SetFwdPerformanceAlgo(fwd ForwardPerformance) {
	c.perfforward = fwd
	c.op.SetMathType(fwd.MathType)
}

//SetBwdDataPerformanceAlgo will sets the convolution algorithm
func (c *Ops) SetBwdDataPerformanceAlgo(bwddata BackDataPerformance) {
	c.perfbackdata = bwddata
	//	c.bwdddesc.SetMathType(bwddata.MathType)
}

//SetBwdFiltPerformanceAlgo will sets the convolution algorithm
func (c *Ops) SetBwdFiltPerformanceAlgo(bwdfilt BackFilterPerformance) {
	c.perfbackfilt = bwdfilt
	//	c.bwdfdesc.SetMathType(bwdfilt.MathType)
}

//GetFwdAlgoPerfList gets a list of forward performance stats
func (c *Ops) GetFwdAlgoPerfList(handle *cudnn.Handler, x, w, y *tensor.Volume, workspace *nvidia.Malloced) ([]ForwardPerformance, error) {
	var fwd gocudnn.ConvolutionFwdFuncs

	maxfwd, err := fwd.GetConvolutionForwardAlgorithmMaxCount(handle.Cudnn())
	if err != nil {
		return nil, err
	}
	fwdlist := make([]gocudnn.ConvFwdAlgoPerformance, 0)
	if workspace == nil {

		fwdlist, err = fwd.FindConvolutionForwardAlgorithm(handle.Cudnn(), x.TD(), w.FD(), c.op, y.TD(), maxfwd)
	} else {
		fwdlist, err = fwd.FindConvolutionForwardAlgorithmEx(handle.Cudnn(), x.TD(), x.Memer(), w.FD(), w.Memer(), c.op, y.TD(), y.Memer(), maxfwd, workspace, workspace.TotalBytes())
	}

	if err != nil {
		return nil, err
	}
	fwper := make([]ForwardPerformance, 0)
	for i := range fwdlist {

		fwper = append(fwper, tofwrdperf(fwdlist[i]))

	}

	return fwper, nil
}

//GetBwdDataAlgoPerfList gets a list of backward data performance stats to set the convolution algo
func (c *Ops) GetBwdDataAlgoPerfList(handle *cudnn.Handler, dx, w, dy *tensor.Volume, workspace *nvidia.Malloced) ([]BackDataPerformance, error) {

	if dx == nil {
		return nil, nil
	}
	var bwd gocudnn.ConvolutionBwdFuncs
	maxbwddata, err := bwd.GetConvolutionBackwardDataAlgorithmMaxCount(handle.Cudnn())
	if err != nil {
		return nil, err
	}
	bwddata := make([]gocudnn.ConvBwdDataAlgoPerformance, 0)
	if workspace == nil {
		bwddata, err = bwd.FindConvolutionBackwardDataAlgorithm(handle.Cudnn(), w.FD(), dy.TD(), c.op, dx.TD(), maxbwddata)
	} else {
		bwddata, err = bwd.FindConvolutionBackwardDataAlgorithmEx(handle.Cudnn(), w.FD(), w.Memer(), dy.TD(), dy.Memer(), c.op, dx.TD(), dx.Memer(), maxbwddata, workspace, workspace.TotalBytes())
	}

	if err != nil {
		fmt.Println("w.FD: ", w.FD(), "dy.TD(): ", dy.TD(), "dx.TD(): ", dx.TD())
		return nil, err
	}
	bwper := make([]BackDataPerformance, 0)
	for i := range bwddata {
		bwper = append(bwper, tobwddperf(bwddata[i]))
		/*
			err = bwddata[i].Status.Error("Not Available")
			if err != nil {

			} else {

			}
		*/

	}
	return bwper, nil
}

//GetBwdFiltAlgoPerfList gets a list of backward filter stats
func (c *Ops) GetBwdFiltAlgoPerfList(handle *cudnn.Handler, x, dw, dy *tensor.Volume, workspace *nvidia.Malloced) ([]BackFilterPerformance, error) {
	var bwd gocudnn.ConvolutionBwdFuncs
	maxbwdfilt, err := bwd.GetConvolutionBackwardFilterAlgorithmMaxCount(handle.Cudnn())
	if err != nil {
		return nil, err
	}
	bwdfilt := make([]gocudnn.ConvBwdFiltAlgoPerformance, 0)
	if workspace == nil {
		bwdfilt, err = bwd.FindConvolutionBackwardFilterAlgorithm(handle.Cudnn(), x.TD(), dy.TD(), c.op, dw.FD(), maxbwdfilt)
	} else {
		bwdfilt, err = bwd.FindConvolutionBackwardFilterAlgorithmEx(handle.Cudnn(), x.TD(), x.Memer(), dy.TD(), dy.Memer(), c.op, dw.FD(), dw.Memer(), maxbwdfilt, workspace, workspace.TotalBytes())
	}

	if err != nil {
		return nil, err
	}
	bwfper := make([]BackFilterPerformance, 0)
	for i := range bwdfilt {
		bwfper = append(bwfper, tobwdfperf(bwdfilt[i]))
		/*
			err = bwdfilt[i].Status.Error("Not Available")
			if err != nil {

			} else {
			}
		*/
	}
	return bwfper, nil
}

//OutputDim will return the dims of what the output tensor should be
func (c *Ops) OutputDim(input *tensor.Volume, filter *tensor.Volume) ([]int32, error) {

	return c.op.GetOutputDims(input.TD(), filter.FD())

}
