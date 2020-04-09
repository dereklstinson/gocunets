package convolution

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/cudart"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
)

func TestOps_OutputDim(t *testing.T) {

	check := func(e error) {
		if e != nil {
			t.Error(e)
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

	convop, err := StageOperation(cmode, dtype, mtype, 1, []int32{2, 2}, []int32{1, 1}, []int32{1, 1})
	check(err)
	function := convop.OutputDim
	outputdims, err := function(vol, filter)
	fmt.Println(outputdims)
	outputdims, err = convop.OutputDim(vol, filter)
	check(err)

	fmt.Println(outputdims)
	t.Error("Checkoutput")
}
