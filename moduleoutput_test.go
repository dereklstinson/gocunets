package gocunets

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"time"
)

func TestCreateSoftmaxModule(t *testing.T) {
	runtime.LockOSThread()
	check := func(e error) {
		if e != nil {
			t.Fatal(e)
		}

	}
	ModuleConcatDebug()
	src := rand.NewSource(time.Now().UnixNano())
	rng := rand.New(src)
	dlist, err := GetDeviceList()
	check(err)
	dev := dlist[0]
	check(dev.Set())
	w := CreateWorker(dev)
	handle := CreateHandle(w, dev, rng.Uint64())
	bldr := CreateBuilder(handle)
	s, err := CreateStream()
	check(err)
	bldr.Frmt.NCHW()
	bldr.Nan.Propigate()
	bldr.Dtype.Float()
	check(handle.SetStream(s))
	fmt.Println("CreateOutputMoudle")
	nbatch := int32(3)
	outputanswers := []float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	}
	//outputanswers := []float32{
	//	1, 0, 0, 0, //0,0
	//	0, 1, 0, 0, //0,1
	//	0, 0, 1, 0, //1,0
	//	0, 0, 0, 1} //1,1
	// 1 0  0 1
	// 0 1  1 0
	// 0 1  1 0
	// 1 0  0 1
	//
	inputanswers := []float32{
		4, -4, -4,
		-4, 4, -4,
		-4, -4, 4,
	}
	//	inputanswers := []float32{
	//		1, 0, 0, 0, //0,0
	//		0, 1, 0, 0, //0,1
	//		0, 0, 1, 0, //1,0
	//		0, 0, 0, 1} //1,1

	inputdims := []int32{nbatch, 3, 1, 1}

	//Create input tensors
	inputx, err := bldr.CreateTensor(inputdims)
	check(err)
	check(inputx.LoadValuesFromSLice(handle.Handler, inputanswers, int32(len(inputanswers))))
	inputdx, err := bldr.CreateTensor(inputdims)
	check(inputdx.SetAll(0))
	check(err)
	outputy, err := bldr.CreateTensor(inputdims)
	check(outputy.SetAll(0))
	check(err)
	outputdy, err := bldr.CreateTensor(inputdims)
	check(err)
	check(outputdy.LoadValuesFromSLice(handle.Handler, outputanswers, int32(len(outputanswers))))
	//softmas is input, back errors, outputfrom softmax, target values
	dmod, err := CreateSoftMaxClassifier(0, bldr, inputx, inputdx, outputy, outputdy)
	check(err)
	fmt.Println("LoadingMemIntoTensor")
	//	inputx.LoadValuesFromSLice(handle.Handler, inputgoslice, int32(len(inputgoslice)))

	check(dmod.PerformError())
	check(bldr.h.Sync())

	//	outputtensory.TogglePrintValueForStringer()
	//	fmt.Println("OutputTensorY:  ", outputtensory)
	//	outputtensory.TogglePrintValueForStringer()

	inputx.TogglePrintValueForStringer()
	fmt.Println("InputTensor: ")
	fmt.Println("*****************************")
	fmt.Println(inputx)
	inputx.TogglePrintValueForStringer()
	outputy.TogglePrintValueForStringer()
	fmt.Println("OutputTensorY: ")
	fmt.Println("*****************************")
	fmt.Println(outputy)
	outputy.TogglePrintValueForStringer()
	outputdy.TogglePrintValueForStringer()
	fmt.Println("Target: ")
	fmt.Println("*****************************")
	fmt.Println(outputdy)
	outputdy.TogglePrintValueForStringer()
	inputdx.TogglePrintValueForStringer()
	fmt.Println("Errors Propagated Backwards: ")
	fmt.Println("*****************************")
	fmt.Println(inputdx)
	inputdx.TogglePrintValueForStringer()
	t.Error("CheckOutput")

	fmt.Println("AverageLoss: ", dmod.GetAverageBatchLoss())
	handle.Close()
}
