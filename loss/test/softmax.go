package main

import (
	"fmt"
	"math"
	"runtime"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/loss"
	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"
)

func main() {
	runtime.LockOSThread()
	dev, err := cudart.GetDevice()
	if err != nil {
		panic(err)
	}
	worker := gocu.NewWorker(dev)
	h := cudnn.CreateHandler(worker, dev, 4)
	var frmt gocudnn.TensorFormat
	var dtype gocudnn.DataType
	dims := []int32{1, 10, 1, 1}
	x, err := layers.CreateTensor(h, frmt.NCHW(), dtype.Float(), dims)
	if err != nil {
		panic(err)
	}
	dx, err := layers.CreateTensor(h, frmt.NCHW(), dtype.Float(), dims)
	if err != nil {
		panic(err)
	}
	y, err := layers.CreateTensor(h, frmt.NCHW(), dtype.Float(), dims)
	if err != nil {
		panic(err)
	}
	dy, err := layers.CreateTensor(h, frmt.NCHW(), dtype.Float(), dims)
	if err != nil {
		panic(err)
	}
	inputdata := []float32{-1, -1, -1, -1, -1, -1, -1, -1, -1, 1}
	targetdata := []float32{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	outputdata := make([]float32, 10)
	dxdata := make([]float32, 10)
	err = x.LoadValuesFromSLice(h, inputdata, (int32)(len(inputdata)))
	if err != nil {
		panic(err)
	}
	err = dy.LoadValuesFromSLice(h, targetdata, (int32)(len(targetdata)))
	if err != nil {
		panic(err)
	}
	sm, err := loss.CreateSoftMax(h, x, dx, y, dy)
	if err != nil {
		panic(err)
	}
	err = h.Sync()
	if err != nil {
		panic(err)
	}
	err = sm.PerformError()
	if err != nil {
		panic(err)
	}
	err = sm.GetTensorY().FillSlice(h, outputdata)
	if err != nil {
		panic(err)
	}
	cpuoutput := softmaxforwardcpu(inputdata)

	for i := range outputdata {

		if !(cpuoutput[i]+.000001 > outputdata[i] && cpuoutput[i]-.000001 < outputdata[i]) {
			panic(fmt.Errorf("cpuoutput[i](%v)!=outputdata[i](%v)", cpuoutput[i], outputdata[i]))
		}
	}
	sm.GetTensorDX().FillSlice(h, dxdata)
	fmt.Println("Cpuoutput", cpuoutput)
	fmt.Println("Outputdata", outputdata)
	fmt.Println("DXdaTA", dxdata)
	fmt.Println("CPUDX", softmaxfinddxcpu(cpuoutput, targetdata))
}

func softmaxforwardcpu(input []float32) (output []float32) {
	output = make([]float32, len(input))
	var denom float32
	for i := range input {
		output[i] = float32(math.Exp(float64(input[i])))
		denom += float32(math.Exp(float64(input[i])))

	}
	for i := range output {
		output[i] /= denom
	}
	return output
}
func softmaxfinddxcpu(output, target []float32) (dx []float32) {
	dx = make([]float32, len(target))
	for i := range output {
		dx[i] = output[i] - target[i]
	}
	return dx
}
