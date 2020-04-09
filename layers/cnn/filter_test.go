//Package cnn contains structs and methods used to do forward, and backward operations for convolution layers
package cnn

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"
)

func evenpadformula(dilation, filter, offset int32) int32 {
	return (((filter - 1) * dilation) + 1 + offset) / 2
}

func TestSetupBasic(t *testing.T) {
	runtime.LockOSThread()
	maj, min, patch, err := gocudnn.GetLibraryVersion()
	if err != nil {
		t.Error(err)
	}
	fmt.Printf("cudnn version{\nmajor: %v,\nminor: %v,\npatch %v\n}\n", maj, min, patch)
	dev, err := cudart.GetDevice()
	if err != nil {
		t.Error(err)
	}
	wrker := gocu.NewWorker(dev)
	handle := cudnn.CreateHandler(wrker, dev, 25)
	var frmt gocudnn.TensorFormat
	var dtype gocudnn.DataType
	var mtype gocudnn.MathType
	var cmode gocudnn.ConvolutionMode
	cmode.CrossCorrelation()
	frmt.NCHW()
	dtype.Float()
	mtype.Default()

	//input specs
	var offset = int32(-2)
	var batch = int32(1)
	var inputhw = int32(8)
	var inputchannels = int32(1)

	//universal filter/convolution properties
	var filterhw = int32(2)
	stride := int32(2)
	var outputchannels = int32(3)
	filterdims := []int32{outputchannels, inputchannels, filterhw, filterhw}
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
	dinputTensor, err := layers.CreateTensor(handle, frmt, dtype, []int32{batch, inputchannels, inputhw, inputhw})
	if err != nil {
		t.Error(err)
	}
	err = dinputTensor.SetValues(handle, 0)
	if err != nil {
		t.Error(err)
	}

	inputTensor1, err := layers.CreateTensor(handle, frmt, dtype, []int32{batch, inputchannels, inputhw, inputhw})
	if err != nil {
		t.Error(err)
	}
	err = inputTensor1.SetValues(handle, 1)
	if err != nil {
		t.Error(err)
	}

	dinputTensor1, err := layers.CreateTensor(handle, frmt, dtype, []int32{batch, inputchannels, inputhw, inputhw})
	if err != nil {
		t.Error(err)
	}
	err = dinputTensor1.SetValues(handle, 0)
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
	pad := evenpadformula(dilation, filterhw, offset)

	layer0, err := SetupBasic(handle, frmt, dtype, mtype, 1, w, dw, b, db, cmode, []int32{pad, pad}, []int32{stride, stride}, []int32{dilation, dilation})
	if err != nil {
		t.Error(err)
	}
	err = layer0.MakeRandom(handle, dinputTensor.Dims())
	if err != nil {
		t.Error(err)
	}

	outputdims0 := layer0.OutputDims(dinputTensor.Dims())
	doutputTensor, err := layers.CreateTensor(handle, frmt, dtype, outputdims0)
	if err != nil {
		t.Error(err)
	}
	err = doutputTensor.SetValues(handle, 2)

	if err != nil {
		t.Error(err)
	}
	outputTensor, err := layers.CreateTensor(handle, frmt, dtype, outputdims0)
	if err != nil {
		t.Error(err)
	}
	err = outputTensor.SetValues(handle, 0)

	if err != nil {
		t.Error(err)
	}
	bwdlago0, err := layer0.GetBwdDataAlgoPerfList(handle, dinputTensor, doutputTensor, nil)
	if err != nil {
		t.Error(err)
	}
	fwdalgo0, err := layer0.GetFwdAlgoPerfList(handle, dinputTensor, doutputTensor, nil)
	if err != nil {
		t.Error(err)
	}
	layer0.SetFwdAlgoPerformance(fwdalgo0[0])

	var wspace *nvidia.Malloced
	if bwdlago0[0].Memory == 0 {
		wspace = nil
	} else {
		wspace, err = nvidia.MallocGlobal(wrker, fwdalgo0[0].Memory)
		if err != nil {
			t.Error(err)
		}
	}
	layer0.SetBwdDataAlgoPerformance(bwdlago0[0])
	var dwspace *nvidia.Malloced
	if bwdlago0[0].Memory == 0 {
		dwspace = nil
	} else {
		dwspace, err = nvidia.MallocGlobal(wrker, bwdlago0[0].Memory)
		if err != nil {
			t.Error(err)
		}
	}
	//fmt.Println(bwdlago0[0])
	//fmt.Println(dinputTensor)
	//fmt.Println(doutputTensor)
	err = layer0.ForwardProp(handle, wspace, inputTensor, outputTensor)
	if err != nil {
		t.Error(err)
	}
	err = layer0.BackPropData(handle, dwspace, dinputTensor, doutputTensor)
	if err != nil {
		t.Error(err)
	}

	dilation1 := int32(3)
	pad1 := evenpadformula(dilation1, filterhw, offset)

	layer1, err := SetupBasic(handle, frmt, dtype, mtype, 1, w, dw, b, db, cmode, []int32{pad1, pad1}, []int32{stride, stride}, []int32{dilation1, dilation1})
	if err != nil {
		t.Error(err)
	}
	err = layer1.MakeRandom(handle, dinputTensor1.Dims())
	if err != nil {
		t.Error(err)
	}

	outputdims1 := layer1.OutputDims(dinputTensor1.Dims())
	fmt.Println("InputDims1", dinputTensor1.Dims())
	fmt.Println("OutputDims1: ", outputdims1)

	doutputTensor1, err := layers.CreateTensor(handle, frmt, dtype, outputdims1)
	if err != nil {
		t.Error(err)
	}
	outputTensor1, err := layers.CreateTensor(handle, frmt, dtype, outputdims1)
	if err != nil {
		t.Error(err)
	}

	err = doutputTensor1.SetValues(handle, 2)
	if err != nil {
		t.Error(err)
	}
	err = outputTensor1.SetValues(handle, 0)
	if err != nil {
		t.Error(err)
	}
	fwdalgo1, err := layer1.GetFwdAlgoPerfList(handle, inputTensor1, outputTensor1, nil)
	if err != nil {
		t.Error(err)
	}
	bwdalgo1, err := layer1.GetBwdDataAlgoPerfList(handle, dinputTensor1, doutputTensor1, nil)
	if err != nil {
		t.Error(err)
	}
	/*	for _, alg := range fwdalgo1 {
		fmt.Println(alg)
	}*/
	dinputTensor1.TogglePrintValueForStringer()
	outputTensor1.TogglePrintValueForStringer()
	for _, algo1 := range fwdalgo1 {
		if algo1.Status.Error("Will Work") == nil {
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

			err = layer1.ForwardProp(handle, wspace1, inputTensor1, outputTensor1)
			if err != nil {
				t.Error(err)
			}

			fmt.Println(outputTensor1)

		} else {
			fmt.Println(algo1)
		}
	}
	for _, algo1 := range bwdalgo1 {

		if algo1.Status.Error("Will Work") == nil {
			layer1.SetBwdDataAlgoPerformance(algo1)
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

			err = layer1.BackPropData(handle, wspace1, dinputTensor1, doutputTensor1)
			if err != nil {
				t.Error(err)
			}

			fmt.Println(dinputTensor1)

		} else {
			fmt.Println(algo1)
		}

	}

	//outputvals := make([]float32, outputTensor.Vol())
	//fmt.Println(outputTensor.Vol())
	//	outputptr, err := cutil.WrapGoMem(outputvals)
	//	if err != nil {
	//		t.Error(err)
	//	}

	//	err = nvidia.Memcpy(outputptr, outputTensor, outputTensor.SIB())
	dinputTensor.TogglePrintValueForStringer()
	//	fmt.Println(layer0)
	fmt.Println("dxtensor dil 1: ")
	fmt.Println(dinputTensor)
	doutputTensor1.TogglePrintValueForStringer()
	fmt.Println(doutputTensor1)
	t.Error("Look at outputs")
	//	fmt.Println(layer1)

	//	fmt.Println(layer1)

}
