package loss

import (
	"fmt"
	"math"
	"runtime"
	"testing"

	gocudnn "github.com/dereklstinson/GoCudnn"

	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
)

func TestCreateSoftMax(t *testing.T) {
	runtime.LockOSThread()
	dev, err := cudart.GetDevice()
	if err != nil {
		t.Error(err)
	}
	worker := gocu.NewWorker(dev)
	h := cudnn.CreateHandler(worker, dev, 4)
	var frmt gocudnn.TensorFormat
	var dtype gocudnn.DataType
	dims := []int32{1, 10, 1, 1}
	x, err := layers.CreateTensor(h, frmt.NCHW(), dtype.Float(), dims)
	if err != nil {
		t.Error(err)
	}
	dx, err := layers.CreateTensor(h, frmt.NCHW(), dtype.Float(), dims)
	if err != nil {
		t.Error(err)
	}
	y, err := layers.CreateTensor(h, frmt.NCHW(), dtype.Float(), dims)
	if err != nil {
		t.Error(err)
	}
	dy, err := layers.CreateTensor(h, frmt.NCHW(), dtype.Float(), dims)
	if err != nil {
		t.Error(err)
	}
	inputdata := []float32{-1, -1, -1, -1, -1, -1, -1, -1, -1, 1}
	targetdata := []float32{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	outputdata := make([]float32, 10)
	err = x.LoadValuesFromSLice(h, inputdata, (int32)(len(inputdata)))
	if err != nil {
		t.Error(err)
	}
	err = dy.LoadValuesFromSLice(h, targetdata, (int32)(len(targetdata)))
	if err != nil {
		t.Error(err)
	}
	sm, err := CreateSoftMax(h)
	if err != nil {
		t.Error(err)
	}
	err = h.Sync()
	if err != nil {
		t.Error(err)
	}
	err = sm.PerformError(x, dx, y, dy)
	if err != nil {
		t.Error(err)
	}
	err = y.FillSlice(h, outputdata)
	if err != nil {
		t.Error(err)
	}
	cpuoutput := softmaxforwardcpu(inputdata)
	cpudx := softmaxfinddxcpu(cpuoutput, targetdata)
	for i := range outputdata {
		if cpuoutput[i] != outputdata[i] {
			t.Errorf("cpuoutput[i](%v)!=outputdata[i](%v)\n", cpuoutput[i], outputdata[i])
		}
	}
	fmt.Println(cpudx)
}

func softmaxforwardcpu(input []float32) (output []float32) {
	output = make([]float32, len(input))
	var denom float32
	for i := range input {
		denom += float32(math.Exp(float64(input[i])))

	}
	for i := range output {
		output[i] = input[i] / denom
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
