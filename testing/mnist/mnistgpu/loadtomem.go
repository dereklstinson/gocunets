package mnistgpu

/*
import (
	"fmt"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/layers"
	"github.com/dereklstinson/gocunets/testing/mnist/dfuncs"
	gocudnn "github.com/dereklstinson/gocudnn"
	"github.com/dereklstinson/gocudnn/gocu"
)

//WithLabels return trainingimages,traininglabels, testimages,testlabels
func WithLabels(handle *cudnn.Handler, batchsize int, frmt cudnn.TensorFormat, dtype cudnn.DataType) ([]*layers.IO, []*layers.IO, []*layers.IO, []*layers.IO) {
	filedirectory := "/home/derek/go/src/github.com/dereklstinson/gocunets/testing/mnist/files/"
	trainingdata, err := dfuncs.LoadMNIST(filedirectory, "train-labels.idx1-ubyte", "train-images.idx3-ubyte")
	cherror(err)
	testingdata, err := dfuncs.LoadMNIST(filedirectory, "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte")
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
		label, err := gocudnn.MakeGoPointer(batchlabelslice)
		cherror(err)
		inpt, err := layers.BuildNetworkInputIO(handle, frmt, dtype, dims(batchsize, 1, 28, 28))
		cherror(err)
		err = inpt.LoadTValues(handle, data)
		cherror(err)
		ansr, err := layers.BuildIO(handle, frmt, dtype, dims(batchsize, 10, 1, 1))
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
		ansr, err := layers.BuildIO(handle, frmt, dtype, dims(batchsize, 10, 1, 1))
		cherror(err)
		err = ansr.LoadDeltaTValues(handle, label)
		cherror(err)
		gputestansdata = append(gputestansdata, ansr)

	}

	fmt.Println("Done Loading Testing To GPU")
	return gputrainingdata, gpuanswersdata, gputestingdata, gputestansdata
}

//Labelbatch contains the labels
type Labelbatch struct {
	labels [][]float32
}

//WithCPULabels return trainingimages,traininglabels, testimages,testlabels
func WithCPULabels(handle *cudnn.Handler, batchsize int, frmt cudnn.TensorFormat, dtype cudnn.DataType) ([]*layers.IO, []Labelbatch) {
	filedirectory := "/home/derek/go/src/github.com/dereklstinson/gocunets/testing/mnist/files/"
	trainingdata, err := dfuncs.LoadMNIST(filedirectory, "train-labels.idx1-ubyte", "train-images.idx3-ubyte")
	cherror(err)
	testingdata, err := dfuncs.LoadMNIST(filedirectory, "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte")
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
	//var gpuanswersdata []*layers.IO
	//var gputestingdata []*layers.IO
	//var gputestansdata []*layers.IO
	//	var cputrainans [][]float32
	//	var cputestans [][]float32
	batchlabels := make([]Labelbatch, 0)
	sizeofdata := 28 * 28

	for i := 0; i < len(trainingdata); { //Counting i inside the j loop, because I don't want to figure out the math
		batchslice := make([]float32, sizeofdata*batchsize)
		//	batchlabelslice := make([]float32, 0)
		var singlebatch Labelbatch
		for j := 0; j < batchsize; j++ {
			for k := 0; k < len(trainingdata[i].Data); k++ {
				batchslice[j*sizeofdata+k] = trainingdata[i].Data[k]
			}
			singlebatch.labels = append(singlebatch.labels, trainingdata[i].Label)
			//	batchlabelslice = append(batchlabelslice, trainingdata[i].Label...)
			i++
		}
		batchlabels = append(batchlabels, singlebatch)
		data, err := gocudnn.MakeGoPointer(batchslice)
		cherror(err)
		//label, err := gocudnn.MakeGoPointer(batchlabelslice)
		//cherror(err)
		inpt, err := layers.BuildNetworkInputIO(handle, frmt, dtype, dims(batchsize, 1, 28, 28))
		cherror(err)
		err = inpt.LoadTValues(handle, data)
		cherror(err)
		//	ansr, err := layers.BuildIO(frmt, dtype, dims(batchsize, 10, 1, 1), memmanaged)
		//	cherror(err)
		//	err = ansr.LoadDeltaTValues(label)
		//cherror(err)
		gputrainingdata = append(gputrainingdata, inpt)
		//	gpuanswersdata = append(gpuanswersdata, ansr)
	}
	fmt.Println("Done Loading Training to GPU")

	fmt.Println("Done Loading Testing To GPU")
	return gputrainingdata, batchlabels
}

func cherror(err error) {
	if err != nil {
		panic(err)
	}
}
func dims(args ...int) []int32 {

	length := len(args)
	x := make([]int32, length)
	for i := 0; i < length; i++ {
		x[i] = int32(args[i])
	}
	return x
}
*/
