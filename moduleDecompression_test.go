package gocunets

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"time"

	"github.com/dereklstinson/GoCudnn/gocu"
)

func TestCreateDecompressionModule(t *testing.T) {
	runtime.LockOSThread()

	//gocudnn.DebugMode()

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
	w := gocu.NewWorker(dev)
	handle := CreateHandle(w, dev, rng.Uint64())
	bldr := CreateBuilder(handle)
	s, err := CreateStream()
	check(err)

	check(handle.SetStream(s))
	fmt.Println("CreateDecompressionModule")
	nbatch := int32(2)
	inputchannels := int32(3)
	inputdims := []int32{nbatch, inputchannels, 8, 8}
	dmod, err := CreateDecompressionModule(0, bldr, nbatch, inputchannels,
		[]int32{3, 2, 1}, []int32{2, 2},
		-2, 1, 0)
	check(err)
	//Create input tensors
	inputx, err := bldr.CreateTensor(inputdims)
	check(err)
	inputdx, err := bldr.CreateTensor(inputdims)
	check(err)
	//SetInputTensors
	dmod.SetTensorX(inputx)
	dmod.SetTensorDX(inputdx)

	//Find output dims
	outputdims, err := dmod.FindOutputDims()
	check(err)
	fmt.Println("inputdims/outputdims", inputx.Dims(), outputdims)
	//Use output dims to find output tensors
	outputtensory, err := bldr.CreateTensor(outputdims)
	check(err)
	outputtensordy, err := bldr.CreateTensor(outputdims)
	check(err)

	//Set  output tensors
	dmod.SetTensorY(outputtensory)
	dmod.SetTensorDY(outputtensordy)

	check(inputx.SetValues(handle.Handler, 1))
	fmt.Println("Start Init Hidden Layers")
	check(dmod.InitHiddenLayers(.000001, .0001))
	fmt.Println("Done Init Hidden Layers")
	fmt.Println("Start Init Workspace")
	check(dmod.InitWorkspace())
	fmt.Println("Done Init Workspace")

	fmt.Println("LoadingMemIntoTensor")
	//	inputx.LoadValuesFromSLice(handle.Handler, inputgoslice, int32(len(inputgoslice)))

	check(outputtensordy.SetValues(handle.Handler, .01))
	fmt.Println("Done Filling Slice")

	check(err)
	fmt.Println("Doing Forward Operation")
	check(dmod.Forward())
	fmt.Println("Doing BackwardOperation")
	check(bldr.h.Sync())
	//	outputtensory.TogglePrintValueForStringer()
	//	fmt.Println("OutputTensorY:  ", outputtensory)
	//	outputtensory.TogglePrintValueForStringer()
	check(dmod.Backward())
	inputdx.TogglePrintValueForStringer()
	fmt.Println("OutputTensordx", inputdx)
	inputdx.TogglePrintValueForStringer()
	t.Error("CheckOutput")
	handle.Close()

}
func TestCreateCompresionModule(t *testing.T) {
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
	w := gocu.NewWorker(dev)
	handle := CreateHandle(w, dev, rng.Uint64())
	bldr := CreateBuilder(handle)
	s, err := CreateStream()
	check(err)

	check(handle.SetStream(s))
	fmt.Println("CreateCompressionModule")
	nbatch := int32(2)
	inputchannels := int32(2)
	inputdims := []int32{nbatch, inputchannels, 8, 8}

	dmod, err := CreateCompressionModule(0, bldr,
		nbatch, inputchannels, []int32{2, 2, 2}, []int32{2, 2}, 2, 1, 0)

	check(err)
	//Create input tensors
	inputx, err := bldr.CreateTensor(inputdims)
	check(err)
	inputdx, err := bldr.CreateTensor(inputdims)
	check(err)

	//SetInputTensors
	dmod.SetTensorX(inputx)
	dmod.SetTensorDX(inputdx)

	//Find output dims
	outputdims, err := dmod.FindOutputDims()
	check(err)
	fmt.Println("inputdims,outputdims", inputdims, outputdims)
	//Use output dims to find output tensors
	outputtensory, err := bldr.CreateTensor(outputdims)
	check(err)
	outputtensordy, err := bldr.CreateTensor(outputdims)
	check(err)

	//Set  output tensors
	dmod.SetTensorY(outputtensory)
	dmod.SetTensorDY(outputtensordy)

	check(inputx.SetValues(handle.Handler, 2))
	fmt.Println("Start Init Hidden Layers")
	check(dmod.InitHiddenLayers(.000001, .0001))
	fmt.Println("Done Init Hidden Layers")
	fmt.Println("Start Init Workspace")
	check(dmod.InitWorkspace())
	fmt.Println("Done Init Workspace")

	fmt.Println("LoadingMemIntoTensor")
	//	inputx.LoadValuesFromSLice(handle.Handler, inputgoslice, int32(len(inputgoslice)))

	check(outputtensordy.SetValues(handle.Handler, .3))
	fmt.Println("Done Filling Slice")

	check(err)
	fmt.Println("Doing Forward Operation")
	check(dmod.Forward())
	fmt.Println("Doing BackwardOperation")
	check(bldr.h.Sync())
	//	outputtensory.TogglePrintValueForStringer()
	//	fmt.Println("OutputTensorY:  ", outputtensory)
	//	outputtensory.TogglePrintValueForStringer()
	check(dmod.Backward())
	inputdx.TogglePrintValueForStringer()
	fmt.Println("OutputTensordx", inputdx)
	inputdx.TogglePrintValueForStringer()
	t.Error("CheckOutput")
	handle.Close()
}

func TestNeurtralModule(t *testing.T) {
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
	w := gocu.NewWorker(dev)
	handle := CreateHandle(w, dev, rng.Uint64())
	bldr := CreateBuilder(handle)
	s, err := CreateStream()
	check(err)

	check(handle.SetStream(s))
	fmt.Println("CreateNeutralModule")
	nbatch := int32(2)
	inputchannels := int32(3)
	inputdims := []int32{nbatch, inputchannels, 4, 4}

	dmod, err := CreateSingleStridedModule(0, bldr,
		nbatch, inputchannels, []int32{3, 2, 1}, []int32{2, 2}, 1, 1, 0, false, false)

	check(err)
	//Create input tensors
	inputx, err := bldr.CreateTensor(inputdims)
	check(err)
	inputdx, err := bldr.CreateTensor(inputdims)
	check(err)

	//SetInputTensors
	dmod.SetTensorX(inputx)
	dmod.SetTensorDX(inputdx)

	//Find output dims
	outputdims, err := dmod.FindOutputDims()
	check(err)
	fmt.Println("inputdims,outputdims", inputdims, outputdims)
	//Use output dims to find output tensors
	outputtensory, err := bldr.CreateTensor(outputdims)
	check(err)
	outputtensordy, err := bldr.CreateTensor(outputdims)
	check(err)

	//Set  output tensors
	dmod.SetTensorY(outputtensory)
	dmod.SetTensorDY(outputtensordy)

	check(inputx.SetValues(handle.Handler, 2))
	fmt.Println("Start Init Hidden Layers")
	check(dmod.InitHiddenLayers(.000001, .0001))
	fmt.Println("Done Init Hidden Layers")
	fmt.Println("Start Init Workspace")
	check(dmod.InitWorkspace())
	fmt.Println("Done Init Workspace")

	fmt.Println("LoadingMemIntoTensor")
	//	inputx.LoadValuesFromSLice(handle.Handler, inputgoslice, int32(len(inputgoslice)))

	check(outputtensordy.SetValues(handle.Handler, .1))
	fmt.Println("Done Filling Slice")

	check(err)
	fmt.Println("Doing Forward Operation")
	check(dmod.Forward())
	fmt.Println("Doing BackwardOperation")
	check(bldr.h.Sync())
	//	outputtensory.TogglePrintValueForStringer()
	//	fmt.Println("OutputTensorY:  ", outputtensory)
	//	outputtensory.TogglePrintValueForStringer()
	check(dmod.Backward())
	for _, dsrc := range dmod.c.deltasrcs {
		dsrc.TogglePrintValueForStringer()
		fmt.Println(dsrc)
		dsrc.TogglePrintValueForStringer()

	}
	inputdx.TogglePrintValueForStringer()
	fmt.Println("OutputTensordx", inputdx)
	inputdx.TogglePrintValueForStringer()
	t.Error("CheckOutput")
	handle.Close()
}
