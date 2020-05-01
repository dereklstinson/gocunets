package mnistgpu

/*
import (
	"fmt"

	"github.com/dereklstinson/gocudnn/gocu"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/layers"
	"github.com/dereklstinson/gocunets/testing/mnist/dfuncs"
	gocudnn "github.com/dereklstinson/gocudnn"
)

//WithLabels11Gan return trainingimages,traininglabels, testimages,testlabels
func WithLabels11Gan(handle *cudnn.Handler, batchsize int, frmt cudnn.TensorFormat, dtype cudnn.DataType, memmanaged bool) ([]*layers.IO, []*layers.IO, []*layers.IO, []*layers.IO) {
	filedirectory := "/home/derek/go/src/github.com/dereklstinson/gocunets/testing/mnist/files/"
	trainingdata, err := dfuncs.LoadMNIST11gan(filedirectory, "train-labels.idx1-ubyte", "train-images.idx3-ubyte")
	cherror(err)
	testingdata, err := dfuncs.LoadMNIST11gan(filedirectory, "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte")
	cherror(err)

	//Normalizing Data
	averagetest := dfuncs.FindAverage(testingdata)
	averagetrain := dfuncs.FindAverage(trainingdata)
	fmt.Println("Finding Average Value")
	averagetotal := ((6.0 * averagetrain) + averagetest) / float32(7)

	fmt.Println("Normalizing Data")
	trainingdata = dfuncs.NormalizeData(trainingdata, averagetotal)
	testingdata = dfuncs.NormalizeData(testingdata, averagetotal)
	fmt.Println("Length of Training Data", len(trainingdata))
	fmt.Println("Length of Testing Data", len(testingdata))

	//Since Data is so small we can load it all into the GPU
	var gputrainingdata []*layers.IO
	var gpuanswersdata []*layers.IO
	var gputestingdata []*layers.IO
	var gputestansdata []*layers.IO

	for i := 0; i < len(trainingdata); { //Counting i inside the j loop, because I don't want to figure out the math
		batchslice := make([]float32, 0)
		batchlabelslice := make([]float32, 0)

		for j := 0; j < batchsize; j++ {
			batchslice = append(batchslice, trainingdata[i].Data...)
			batchlabelslice = append(batchlabelslice, trainingdata[i].Label...)
			i++
		}

		data, err := gocu.MakeGoMem(batchslice)
		cherror(err)
		label, err := gocu.MakeGoMem(batchlabelslice)
		cherror(err)
		inpt, err := layers.BuildNetworkInputIO(handle, frmt, dtype, dims(batchsize, 1, 28, 28))
		cherror(err)
		err = inpt.LoadTValues(handle, data)
		cherror(err)
		ansr, err := layers.BuildIO(handle, frmt, dtype, dims(batchsize, 11, 1, 1))
		cherror(err)
		err = ansr.LoadDeltaTValues(handle, label)
		cherror(err)
		gputrainingdata = append(gputrainingdata, inpt)
		gpuanswersdata = append(gpuanswersdata, ansr)
	}
	fmt.Println("Done Loading Training to GPU")

	for i := 0; i < len(testingdata); {
		batchslice := make([]float32, 0)
		batchlabelslice := make([]float32, 0)
		for j := 0; j < batchsize; j++ {
			batchslice = append(batchslice, testingdata[i].Data...)
			batchlabelslice = append(batchlabelslice, testingdata[i].Label...)
			i++
		}
		data, err := gocudnn.MakeGoPointer(batchslice)
		cherror(err)
		label, err := gocudnn.MakeGoPointer(batchlabelslice)
		cherror(err)
		inpt, err := layers.BuildNetworkInputIO(handle, frmt, dtype, dims(batchsize, 1, 28, 28))
		cherror(err)
		err = inpt.LoadTValues(handle, data)
		cherror(err)
		gputestingdata = append(gputestingdata, inpt)
		ansr, err := layers.BuildIO(handle, frmt, dtype, dims(batchsize, 11, 1, 1))
		cherror(err)
		err = ansr.LoadDeltaTValues(handle, label)
		cherror(err)
		gputestansdata = append(gputestansdata, ansr)

	}

	fmt.Println("Done Loading Testing To GPU")
	return gputrainingdata, gpuanswersdata, gputestingdata, gputestansdata
}
*/
