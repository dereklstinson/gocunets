package cnntranspose

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"

	gocudnn "github.com/dereklstinson/GoCudnn"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"
)

func TestSetupBasic(t *testing.T) {
	runtime.LockOSThread()
	dev, err := cudart.GetDevice()
	if err != nil {
		t.Error(err)
	}
	wrker := gocu.NewWorker(dev)
	handle := cudnn.CreateHandler(wrker, dev, 3)
	var frmt gocudnn.TensorFormat
	var dtype gocudnn.DataType
	var mtype gocudnn.MathType
	var cmode gocudnn.ConvolutionMode
	frmt.NCHW()
	dtype.Float()
	mtype.Default()
	//input specs
	var batch = int32(2)
	var inputhw = int32(4)
	var inputchannels = int32(3)

	//universal filter/convolution properties
	var filterhw = int32(2)
	stride := int32(2)
	var outputchannels = int32(4)
	filterdims := []int32{inputchannels, outputchannels, filterhw, filterhw}
	biasdims := []int32{1, outputchannels, 1, 1}

	//Input Tensor
	inputTensor, err := layers.CreateTensor(handle, frmt, dtype, []int32{batch, inputchannels, inputhw, inputhw})
	if err != nil {
		t.Error(err)
	}
	err = inputTensor.SetValues(handle, 1)
	if err != nil {
		t.Error(err)
	}

	//w,dw,b,db
	w, err := layers.CreateTensor(handle, frmt, dtype, filterdims)
	if err != nil {
		t.Error(err)
	}
	dw, err := layers.CreateTensor(handle, frmt, dtype, filterdims)
	if err != nil {
		t.Error(err)
	}
	b, err := layers.CreateTensor(handle, frmt, dtype, biasdims)
	if err != nil {
		t.Error(err)
	}
	db, err := layers.CreateTensor(handle, frmt, dtype, biasdims)
	if err != nil {
		t.Error(err)
	}
	dilation := int32(1)
	pad := evenpadformula(dilation, filterhw)

	layer0, err := SetupBasic(handle, frmt, dtype, mtype, 1, w, dw, b, db, cmode, []int32{pad, pad}, []int32{stride, stride}, []int32{dilation, dilation})
	if err != nil {
		t.Error(err)
	}
	err = layer0.MakeRandom(handle, inputTensor.Dims())
	if err != nil {
		t.Error(err)
	}

	outputdims0 := layer0.OutputDims(inputTensor.Dims())
	outputTensor, err := layers.CreateTensor(handle, frmt, dtype, outputdims0)
	if err != nil {
		t.Error(err)
	}
	fwdalgo, err := layer0.GetFwdAlgoPerfList(handle, inputTensor, outputTensor, nil)
	if err != nil {
		t.Error(err)
	}
	layer0.SetFwdAlgoPerformance(fwdalgo[0])
	var wspace *nvidia.Malloced
	if fwdalgo[0].Memory == 0 {
		wspace = nil
	} else {
		wspace, err = nvidia.MallocGlobal(wrker, fwdalgo[0].Memory)
		if err != nil {
			t.Error(err)
		}
	}

	fmt.Println(fwdalgo[0])
	fmt.Println(inputTensor)
	fmt.Println(outputTensor)

	dilation1 := int32(3)
	pad1 := evenpadformula(dilation1, filterhw)

	layer1, err := SetupBasic(handle, frmt, dtype, mtype, 1, w, dw, b, db, cmode, []int32{pad1, pad1}, []int32{stride, stride}, []int32{dilation1, dilation1})
	if err != nil {
		t.Error(err)
	}
	err = layer1.MakeRandom(handle, inputTensor.Dims())
	if err != nil {
		t.Error(err)
	}

	outputdims1 := layer1.OutputDims(inputTensor.Dims())
	fmt.Println("InputDims1", inputTensor.Dims())
	fmt.Println("OutputDims1: ", outputdims1)

	outputTensor1, err := layers.CreateTensor(handle, frmt, dtype, outputdims1)

	if err != nil {
		t.Error(err)
	}
	fwdalgo1, err := layer1.GetFwdAlgoPerfList(handle, inputTensor, outputTensor1, nil)
	if err != nil {
		t.Error(err)
	}
	/*	for _, alg := range fwdalgo1 {
		fmt.Println(alg)
	}*/
	algo1 := fwdalgo1[0]
	layer1.SetFwdAlgoPerformance(algo1)
	var wspace1 *nvidia.Malloced
	if algo1.Memory == 0 {
		wspace1 = nil
	} else {
		wspace1, err = nvidia.MallocGlobal(wrker, algo1.Memory)
		if err != nil {
			t.Error(err)
		}
	}
	fmt.Println(algo1)
	err = layer0.ForwardProp(handle, wspace, inputTensor, outputTensor)
	if err != nil {
		t.Error(err)
	}
	err = layer1.ForwardProp(handle, wspace1, inputTensor, outputTensor1)
	if err != nil {
		t.Error(err)
	}

	//outputvals := make([]float32, outputTensor.Vol())
	//fmt.Println(outputTensor.Vol())
	//	outputptr, err := cutil.WrapGoMem(outputvals)
	//	if err != nil {
	//		t.Error(err)
	//	}

	//	err = nvidia.Memcpy(outputptr, outputTensor, outputTensor.SIB())
	outputTensor.TogglePrintValueForStringer()
	fmt.Println("Outputtensor 0", outputTensor)

	outputTensor1.TogglePrintValueForStringer()
	//	fmt.Println(layer1)
	fmt.Println("OutputTensor 1", outputTensor1)
	t.Error("look at outputs")

}

func evenpadformula(dilation, filter int32) int32 {
	return (((filter - 1) * dilation) - 1) / 2
}
