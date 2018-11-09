package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/pooling"
	"github.com/dereklstinson/GoCuNets/layers/softmax"
	"github.com/dereklstinson/GoCuNets/testing/mnist/mnistgpu"
	"github.com/dereklstinson/GoCuNets/trainer"
	"github.com/dereklstinson/GoCudnn" //	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	//	gocudnn "github.com/dereklstinson/GoCudnn"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	//	savelocationforimages := "/home/derek/Desktop/GANMNIST/"
	//	imagenames := "MNIST"
	trainingkernellocation := "/home/derek/go/src/github.com/dereklstinson/GoCudnn/kernels/"
	gocudnn.Cuda{}.LockHostThread()
	var cuda gocudnn.Cuda
	devices, err := cuda.GetDeviceList()
	cherror(err)
	devicenum := len(devices)
	fmt.Println("Number of Devices:", devicenum)
	device := devices[0]
	if len(devices) == 2 {
		device = devices[1]
	}

	err = device.Set()
	cherror(err)
	handle := cudnn.CreateHandler(device, trainingkernellocation)
	stream, err := gocudnn.Cuda{}.CreateBlockingStream()
	cherror(handle.SetStream(stream))

	var dtypeflags cudnn.DataTypeFlag
	var fmtflags cudnn.TensorFormatFlag
	frmt := fmtflags.NCHW()
	dtype := dtypeflags.Float()
	CMode := convolution.Flags().Mode.CrossCorrelation() //.CrossCorrelation()

	Pmode := gocudnn.PoolingModeFlag{}.Max()
	NanProp := gocudnn.PropagationNANFlag{}.PropagateNan()
	memmanaged := true

	in := dims
	filter := dims
	padding := dims
	stride := dims
	dilation := dims

	/*
	   Lets go ahead and start loading the training data

	*/
	//asdfas

	batchsize := 20 // how many forward and backward runs before updating weights.

	gputrainingdata, gpuanswersdata, gputestingdata, gputestansdata := mnistgpu.WithLabels(batchsize, frmt, dtype, memmanaged)
	batchnum := len(gputrainingdata)
	testbatchnum := len(gputestingdata)

	//AMode := gocudnn.ActivationModeFlag{}.Relu()
	network := gocunets.CreateNetwork()
	//Setting Up Network
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle, frmt, dtype, in(batchsize, 1, 28, 28), filter(20, 1, 5, 5), CMode, padding(2, 2), stride(1, 1), dilation(1, 1), memmanaged),
	)
	network.AddLayer( //activation
		activation.Leaky(handle),
	)
	network.AddLayer( //pooling
		pooling.SetupDims(Pmode, NanProp, 4, filter(2, 2), padding(0, 0), stride(2, 2), memmanaged),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle, frmt, dtype, in(batchsize, 20, 14, 14), filter(20, 20, 5, 5), CMode, padding(2, 2), stride(1, 1), dilation(1, 1), memmanaged),
	)
	network.AddLayer( //activation
		activation.Leaky(handle),
	)
	network.AddLayer( //pooling
		pooling.SetupDims(Pmode, NanProp, 4, filter(2, 2), padding(0, 0), stride(2, 2), memmanaged),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle, frmt, dtype, in(batchsize, 20, 7, 7), filter(20, 20, 3, 3), CMode, padding(1, 1), stride(2, 2), dilation(1, 1), memmanaged),
	)
	network.AddLayer( //activation
		activation.Leaky(handle),
	)

	network.AddLayer( //convolution
		cnn.SetupDynamic(handle, frmt, dtype, in(batchsize, 20, 4, 4), filter(10, 20, 4, 4), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
		//fcnn.CreateFromshapeNoOut(handle.Cudnn(), 10, in(batchsize, 20, 4, 4), memmanaged, dtype, frmt),
	)
	network.AddLayer( //softmaxoutput
		softmax.BuildNoErrorChecking(), nil,
	)
	cherror(network.DynamicHidden())
	//cherror(network.StaticHidden(handle))
	numoftrainers := network.TrainersNeeded()
	fmt.Println("Number of Trainers:", numoftrainers)
	decay1, decay2 := float32(0.000001), float32(0.0001)
	wtrainer := make([]trainer.Trainer, numoftrainers)
	btrainer := make([]trainer.Trainer, numoftrainers)
	for i := 0; i < numoftrainers; i++ {
		wtrainer[i], btrainer[i], err = trainer.SetupAdamWandB(handle.XHandle(), decay1, decay2, int32(batchsize))
		cherror(err)

	}

	network.LoadTrainers(handle, wtrainer, btrainer)
	//imager, err := imaging.MakeImager(handle.XHandle())
	epochs := 50
	//	inputslicefromgpumem := make([]float32, 28*28)
	for k := 0; k < epochs; k++ {

		for j := 0; j < batchnum; j++ { //I add the j++ at the end of this
			//		fmt.Println("Epoch:", k, "Batch:", j)
			//	cuda.CtxSynchronize()
			cherror(stream.Sync())
			cherror(network.ForwardProp(handle, nil, gputrainingdata[j], gpuanswersdata[j]))
			cherror(stream.Sync())
			cherror(network.BackPropFilterData(handle, nil, gputrainingdata[j], gpuanswersdata[j]))
			cherror(stream.Sync())
			cherror(network.UpdateWeights(handle, batchsize))
			cherror(stream.Sync())

		}

		netoutput := make([][]float32, testbatchnum)
		desiredoutput := make([][]float32, testbatchnum)
		for j := 0; j < testbatchnum; j++ {

			cherror(network.ForwardProp(handle, nil, gputestingdata[j], gputestansdata[j]))
			cherror(stream.Sync())

			netoutput[j] = make([]float32, 10*batchsize)

			desiredoutput[j] = make([]float32, 10*batchsize)

			cherror(gputestansdata[j].T().Memer().FillSlice(netoutput[j]))
			cherror(stream.Sync())
			cherror(gputestansdata[j].DeltaT().Memer().FillSlice(desiredoutput[j]))
			cherror(stream.Sync())
		}
		cherror(stream.Sync())
		func(netoutput [][]float32, desiredoutput [][]float32, k int, testbatchnum int, batchsize int) {
			percent, loss := epocoutputchecker(netoutput, desiredoutput, testbatchnum, batchsize, 10)
			fmt.Printf("Epoch Percent Correct: %-0.3f		 Epoch Loss: %-0.3f              Epoch Number: %d\n", percent, loss, k)

		}(netoutput, desiredoutput, k, testbatchnum, batchsize)

	}

	gocudnn.Cuda{}.UnLockHostThread()
	cherror(device.Reset())

}
func printoutput(numofans, batchsize int, input []float32) {
	for i := 0; i < batchsize; i++ {
		for j := 0; j < numofans; j++ {
			fmt.Printf("%-0.2f ", input[i*numofans+j])
		}
		fmt.Printf("\n ")
	}
}
func epocoutputchecker(actual, desired [][]float32, batchtotal, batchsize, classificationsize int) (float64, float64) {
	var batchloss float64
	var percent float64
	for i := 0; i < batchtotal; i++ {
		perc, batch := batchoutputchecker(actual[i], desired[i], batchsize, classificationsize)
		batchloss += batch
		percent += perc
	}
	return percent / float64(batchtotal), batchloss / float64(batchtotal)

}
func batchoutputchecker(actual, desired []float32, batchsize, classificationsize int) (float64, float64) {
	var batchloss float64
	var percent float64
	var position int
	//	delta := float64(-math.Log(float64(output.SoftOutputs[i]))) * desiredoutput[i]
	for i := 0; i < batchsize; i++ {

		maxvalue := float32(0.0)
		ipos := i * classificationsize
		for j := 0; j < classificationsize; j++ {
			ijpos := ipos + j
			if maxvalue < actual[ijpos] {

				maxvalue = actual[ijpos]
				position = ijpos

			}
			if desired[ijpos] != 0 {
				value := (-math.Log(float64(actual[ijpos])))
				if math.IsInf(float64(value), 0) == true {
					fmt.Println("Output Value: ", value)
				}
				batchloss += value
			}

		}
		percent += float64(desired[position])

	}
	if math.IsNaN(float64(batchloss)) == true {
		panic("reach NAN")
	}

	return percent / float64(batchsize), batchloss / float64(batchsize)
}

func dims(args ...int) []int32 {

	length := len(args)
	x := make([]int32, length)
	for i := 0; i < length; i++ {
		x[i] = int32(args[i])
	}
	return x
}
func cherror(input error) {
	if input != nil {
		fmt.Println("***************************")
		panic(input)

	}
}
func getsize(dims []int32) int32 {
	mult := int32(1)
	for i := range dims {
		mult *= dims[i]
	}
	return mult

}
