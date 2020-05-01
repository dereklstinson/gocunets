package main

import (
	"fmt"
	"runtime"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/convolution"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/gocudnn"
	"github.com/dereklstinson/gocudnn/cudart"
)

func main() {

	check := func(e error) {
		if e != nil {
			panic(e)
		}
	}
	runtime.LockOSThread()
	dev, err := cudart.GetDevice()
	check(err)
	handle := cudnn.CreateHandler(dev)
	var frmt gocudnn.TensorFormat
	var dtype gocudnn.DataType
	frmt.NCHW()
	dtype.Float()
	tensordims := []int32{12, 12, 12, 12}

	vol, err := tensor.Build(handle, frmt, dtype, tensordims)
	check(err)
	filterdims := []int32{5, 12, 3, 3}
	filter, err := tensor.Build(handle, frmt, dtype, filterdims)
	check(err)
	var cmode gocudnn.ConvolutionMode
	var mtype gocudnn.MathType
	cmode.CrossCorrelation()
	mtype.Default()

	convop, err := convolution.StageOperation(cmode, dtype, mtype, 1, []int32{2, 2}, []int32{1, 1}, []int32{1, 1})
	check(err)
	dims, err := convop.OutputDim(vol, filter)
	check(err)
	fmt.Println(dims)
}
