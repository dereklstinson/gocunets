package deconvolution

import (
	"fmt"

	gocudnn "github.com/dereklstinson/gocudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/tensor"
)

//ForwardPerformance is used to find the forward performance of a deconvolution algorithm
type ForwardPerformance gocudnn.DeConvFwdAlgoPerformance

func (f ForwardPerformance) String() string {
	return gocudnn.DeConvFwdAlgoPerformance(f).String()
}

//Print will print the forward performance to console
func (f ForwardPerformance) Print() {
	f.Print()
}
func tofwrdperf(x gocudnn.DeConvFwdAlgoPerformance) ForwardPerformance {
	return ForwardPerformance{
		Algo:        x.Algo,
		Status:      x.Status,
		Time:        x.Time,
		Memory:      x.Memory,
		Determinism: x.Determinism,
		MathType:    x.MathType,
	}
}

//BackDataPerformance is used to find the backward data performance of a deconvolution algorithm
type BackDataPerformance gocudnn.DeConvBwdDataAlgoPerformance

func (b BackDataPerformance) String() string {
	return gocudnn.DeConvBwdDataAlgoPerformance(b).String()
}

//Print will print forward performance to console
func (b BackDataPerformance) Print() {
	b.Print()
}

func tobwddperf(x gocudnn.DeConvBwdDataAlgoPerformance) BackDataPerformance {
	return BackDataPerformance{
		Algo:        x.Algo,
		Status:      x.Status,
		Time:        x.Time,
		Memory:      x.Memory,
		Determinism: x.Determinism,
		MathType:    x.MathType,
	}
}

type BackFilterPerformance gocudnn.DeConvBwdFiltAlgoPerformance

func (b BackFilterPerformance) String() string {
	return gocudnn.DeConvBwdFiltAlgoPerformance(b).String()
}

func tobwdfperf(x gocudnn.DeConvBwdFiltAlgoPerformance) BackFilterPerformance {
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
func (c *Ops) SetFwdPerformanceAlgo(fwd ForwardPerformance) error {
	c.perfforward = fwd
	return c.op.SetMathType(fwd.MathType)
}

//SetBwdDataPerformanceAlgo will sets the convolution algorithm
func (c *Ops) SetBwdDataPerformanceAlgo(bwddata BackDataPerformance) error {
	c.perfbackdata = bwddata
	return c.op.SetMathType(bwddata.MathType)
}

//SetBwdFiltPerformanceAlgo will sets the convolution algorithm
func (c *Ops) SetBwdFiltPerformanceAlgo(bwdfilt BackFilterPerformance) error {
	c.perfbackfilt = bwdfilt
	return c.op.SetMathType(bwdfilt.MathType)
}

//GetFwdAlgoPerfList gets a list of forward performance stats
func (c *Ops) GetFwdAlgoPerfList(handle *cudnn.Handler, x, w, y *tensor.Volume, workspace *nvidia.Malloced) ([]ForwardPerformance, error) {
	var err error
	fwdlist := make([]gocudnn.DeConvFwdAlgoPerformance, 0)
	if workspace == nil {

		fwdlist, err = c.op.GetForwardAlgorithmV7(handle.Cudnn(), x.TD(), w.FD(), y.TD())
	} else {
		fwdlist, err = c.op.FindForwardAlgorithmEx(handle.Cudnn(), x.TD(), x, w.FD(), w, y.TD(), y, workspace, workspace.SIB())
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

	var err error
	bwddata := make([]gocudnn.DeConvBwdDataAlgoPerformance, 0)
	if workspace == nil {
		bwddata, err = c.op.FindBackwardDataAlgorithm(handle.Cudnn(), w.FD(), dy.TD(), dx.TD())
		//	bwddata, err = c.op.GetBackwardDataAlgorithmV7(handle.Cudnn(), w.FD(), dy.TD(), dx.TD())
		//fmt.Println("Did Algo7")
	} else {
		bwddata, err = c.op.FindBackwardDataAlgorithmEx(handle.Cudnn(), w.FD(), w, dy.TD(), dy, dx.TD(), dx, workspace, workspace.SIB())
		//	fmt.Println("Did regular algo")
	}

	if err != nil {
		fmt.Println("w.FD: ", w.FD(), "dy.TD(): ", dy.TD(), "dx.TD(): ", dx.TD())
		return nil, err
	}
	bwper := make([]BackDataPerformance, 0)
	for i := range bwddata {
		bwper = append(bwper, tobwddperf(bwddata[i]))

	}
	return bwper, nil
}

//GetBwdFiltAlgoPerfList gets a list of backward filter stats
func (c *Ops) GetBwdFiltAlgoPerfList(handle *cudnn.Handler, x, dw, dy *tensor.Volume, workspace *nvidia.Malloced) ([]BackFilterPerformance, error) {
	var err error

	bwdfilt := make([]gocudnn.DeConvBwdFiltAlgoPerformance, 0)
	if workspace == nil {
		bwdfilt, err = c.op.GetBackwardFilterAlgorithmV7(handle.Cudnn(), x.TD(), dy.TD(), dw.FD())
	} else {
		bwdfilt, err = c.op.FindBackwardFilterAlgorithmEx(handle.Cudnn(), x.TD(), x, dy.TD(), dy, dw.FD(), dw, workspace, workspace.SIB())
	}

	if err != nil {
		return nil, err
	}
	bwfper := make([]BackFilterPerformance, 0)
	for i := range bwdfilt {
		bwfper = append(bwfper, tobwdfperf(bwdfilt[i]))

	}
	return bwfper, nil
}

//OutputDim will return the dims of what the output tensor should be
func (c *Ops) OutputDim(input *tensor.Volume, filter *tensor.Volume) (dims []int32, err error) {
	if input == nil {
		panic("input nil")
	}
	if filter == nil {
		panic("filter nil")
	}
	if c.op == nil {
		panic("op is nil")
	}

	dims, err = c.op.GetOutputDims(input.TD(), filter.FD())
	return dims, err

}
