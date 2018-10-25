package mnistgpu

import (
	"fmt"

	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/testing/mnist/dfuncs"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//MNISTGpuLabels return trainingimages,traininglabels, testimages,testlabels
func MNISTGpuLabels(batchsize int, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, memmanaged bool) ([]*layers.IO, []*layers.IO, []*layers.IO, []*layers.IO) {
	filedirectory := "/home/derek/go/src/github.com/dereklstinson/GoCuNets/testing/mnist/files/"
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

		data, err := gocudnn.MakeGoPointer(batchslice)
		cherror(err)
		label, err := gocudnn.MakeGoPointer(batchlabelslice)
		cherror(err)
		inpt, err := layers.BuildNetworkInputIO(frmt, dtype, dims(batchsize, 1, 28, 28), memmanaged)
		cherror(err)
		err = inpt.LoadTValues(data)
		cherror(err)
		ansr, err := layers.BuildIO(frmt, dtype, dims(batchsize, 10, 1, 1), memmanaged)
		cherror(err)
		err = ansr.LoadDeltaTValues(label)
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
		inpt, err := layers.BuildNetworkInputIO(frmt, dtype, dims(batchsize, 1, 28, 28), memmanaged)
		cherror(err)
		err = inpt.LoadTValues(data)
		cherror(err)
		gputestingdata = append(gputestingdata, inpt)
		ansr, err := layers.BuildIO(frmt, dtype, dims(batchsize, 10, 1, 1), memmanaged)
		cherror(err)
		err = ansr.LoadDeltaTValues(label)
		cherror(err)
		gputestansdata = append(gputestansdata, ansr)

	}

	fmt.Println("Done Loading Testing To GPU")
	return gputrainingdata, gpuanswersdata, gputestingdata, gputestansdata
}

//MNISTGpuNoLabel return trainingimages,traininglabels, testimages,testlabels
func MNISTGpuNoLabel(batchsize int, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, memmanaged bool) ([]*layers.IO, []*layers.IO) {
	filedirectory := "/home/derek/go/src/github.com/dereklstinson/GoCuNets/testing/mnist/files/"
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
	var gputestingdata []*layers.IO
	//var gputestansdata []*layers.IO
	//	var cputrainans [][]float32
	//	var cputestans [][]float32
	for i := 0; i < len(trainingdata); { //Counting i inside the j loop, because I don't want to figure out the math
		batchslice := make([]float32, 0)
		//	batchlabelslice := make([]float32, 0)

		for j := 0; j < batchsize; j++ {
			batchslice = append(batchslice, trainingdata[i].Data...)

			//	batchlabelslice = append(batchlabelslice, trainingdata[i].Label...)
			i++
		}

		data, err := gocudnn.MakeGoPointer(batchslice)
		cherror(err)
		//label, err := gocudnn.MakeGoPointer(batchlabelslice)
		//cherror(err)
		inpt, err := layers.BuildNetworkInputIO(frmt, dtype, dims(batchsize, 1, 28, 28), memmanaged)
		cherror(err)
		err = inpt.LoadTValues(data)
		cherror(err)
		//	ansr, err := layers.BuildIO(frmt, dtype, dims(batchsize, 10, 1, 1), memmanaged)
		//	cherror(err)
		//	err = ansr.LoadDeltaTValues(label)
		//cherror(err)
		gputrainingdata = append(gputrainingdata, inpt)
		//	gpuanswersdata = append(gpuanswersdata, ansr)
	}
	fmt.Println("Done Loading Training to GPU")

	for i := 0; i < len(testingdata); {
		batchslice := make([]float32, 0)
		//	batchlabelslice := make([]float32, 0)
		for j := 0; j < batchsize; j++ {
			batchslice = append(batchslice, testingdata[i].Data...)
			//	batchlabelslice = append(batchlabelslice, testingdata[i].Label...)
			i++
		}
		data, err := gocudnn.MakeGoPointer(batchslice)
		cherror(err)
		//	label, err := gocudnn.MakeGoPointer(batchlabelslice)
		cherror(err)
		inpt, err := layers.BuildNetworkInputIO(frmt, dtype, dims(batchsize, 1, 28, 28), memmanaged)
		cherror(err)
		err = inpt.LoadTValues(data)
		cherror(err)
		gputestingdata = append(gputestingdata, inpt)
		//ansr, err := layers.BuildIO(frmt, dtype, dims(batchsize, 10, 1, 1), memmanaged)
		//cherror(err)
		//err = ansr.LoadDeltaTValues(label)
		//	cherror(err)
		//		gputestansdata = append(gputestansdata, ansr)

	}

	fmt.Println("Done Loading Testing To GPU")
	return gputrainingdata, gputestingdata
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
