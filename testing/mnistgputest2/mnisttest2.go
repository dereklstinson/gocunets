package main

import (
	"fmt"
	"math"

	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/gocudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/fcnn"
	"github.com/dereklstinson/GoCuNets/layers/pooling"
	"github.com/dereklstinson/GoCuNets/layers/softmax"
	"github.com/dereklstinson/GoCuNets/testing/mnist/mnistgpu"
	"github.com/dereklstinson/GoCuNets/trainer"
	"github.com/dereklstinson/GoCudnn" //	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	//	gocudnn "github.com/dereklstinson/GoCudnn"
)

func main() {

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
	handle := gocunets.CreateHandle(device, trainingkernellocation)
	stream, err := gocudnn.Cuda{}.CreateBlockingStream()
	cherror(handle.SetStream(stream))

	var dtypeflags gocudnn.DataTypeFlag
	var fmtflags gocudnn.TensorFormatFlag
	frmt := fmtflags.NCHW()
	dtype := dtypeflags.Float()
	CMode := convolution.Flags().Mode.CrossCorrelation() //.CrossCorrelation()
	//	AMode := gocudnn.ActivationModeFlag{}.Relu()
	Pmode := gocudnn.PoolingModeFlag{}.Max()
	NanProp := gocudnn.PropagationNANFlag{}.NotPropagateNan()
	memmanaged := true
	//	dims := gocudnn.Tensor.Shape
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

	gputrainingdata, gpuanswersdata, gputestingdata, gputestansdata := mnistgpu.MNISTGpuLabels(batchsize, frmt, dtype, memmanaged)
	batchnum := len(gputrainingdata)
	testbatchnum := len(gputestingdata)
	network := gocunets.CreateNetwork()
	AMode := gocudnn.ActivationModeFlag{}.Relu()

	//Setting Up Network
	network.AddLayer( //convolution
		cnn.LayerSetupV2(handle.Cudnn(), frmt, dtype, in(batchsize, 1, 28, 28), filter(20, 1, 5, 5), CMode, padding(2, 2), stride(1, 1), dilation(1, 1), 0, false, memmanaged),
	)
	network.AddLayer( //activation
		activation.SetupNoOut(AMode, memmanaged),
	)
	network.AddLayer( //pooling
		pooling.SetupDims(Pmode, NanProp, 4, filter(2, 2), padding(0, 0), stride(2, 2), memmanaged),
	)
	network.AddLayer( //convolution
		cnn.LayerSetupV2(handle.Cudnn(), frmt, dtype, in(batchsize, 20, 14, 14), filter(20, 20, 5, 5), CMode, padding(2, 2), stride(1, 1), dilation(1, 1), 0, false, memmanaged),
	)
	network.AddLayer( //activation
		activation.SetupNoOut(AMode, memmanaged),
	)
	network.AddLayer( //pooling
		pooling.SetupDims(Pmode, NanProp, 4, filter(2, 2), padding(0, 0), stride(2, 2), memmanaged),
	)
	network.AddLayer( //convolution
		cnn.LayerSetupV2(handle.Cudnn(), frmt, dtype, in(batchsize, 20, 7, 7), filter(20, 20, 3, 3), CMode, padding(1, 1), stride(2, 2), dilation(1, 1), 0, false, memmanaged),
	)
	network.AddLayer( //activation
		activation.SetupNoOut(AMode, memmanaged),
	)

	network.AddLayer( //convolution
		fcnn.CreateFromshapeNoOut(handle.Cudnn(), 10, in(batchsize, 20, 4, 4), memmanaged, dtype, frmt),
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
		wtrainer[i], btrainer[i], err = trainer.SetupAdamWandB(handle.XHandle(), decay1, decay2, batchsize)
		cherror(err)

	}

	network.LoadTrainers(handle, wtrainer, btrainer)

	epochs := 50
	//	inputslicefromgpumem := make([]float32, 28*28)
	for k := 0; k < epochs; k++ {

		for j := 0; j < batchnum; j++ { //I add the j++ at the end of this
			//		fmt.Println("Epoch:", k, "Batch:", j)
			//	cuda.CtxSynchronize()

			cherror(network.ForwardProp(handle, nil, gputrainingdata[j], gpuanswersdata[j]))
			cherror(network.BackProp(handle, nil, gputrainingdata[j], gpuanswersdata[j]))
			cherror(network.UpdateWeights(handle, batchsize))

		}

		netoutput := make([][]float32, testbatchnum)
		desiredoutput := make([][]float32, testbatchnum)
		for j := 0; j < testbatchnum; j++ {
			cherror(network.ForwardProp(handle, nil, gputestingdata[j], gputestansdata[j]))
			cherror(stream.Sync())

			netoutput[j] = make([]float32, 10*batchsize)
			desiredoutput[j] = make([]float32, 10*batchsize)
			err = gputestansdata[j].T().Memer().FillSlice(netoutput[j])
			cherror(err)
			err = gputestansdata[j].DeltaT().Memer().FillSlice(desiredoutput[j])
			cherror(err)

		}

		go func(netoutput [][]float32, desiredoutput [][]float32, k int, testbatchnum int, batchsize int) {
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
func epocoutputchecker(actual, desired [][]float32, batchtotal, batchsize, classificationsize int) (float32, float32) {
	var batchloss float32
	var percent float32
	for i := 0; i < batchtotal; i++ {
		perc, batch := batchoutputchecker(actual[i], desired[i], batchsize, classificationsize)
		batchloss += batch
		percent += perc
	}
	return percent / float32(batchtotal), batchloss / float32(batchtotal)

}
func batchoutputchecker(actual, desired []float32, batchsize, classificationsize int) (float32, float32) {
	var batchloss float32
	var percent float32
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

				batchloss += float32(-math.Log(float64(actual[ijpos])))
			}

		}
		percent += desired[position]

	}
	if math.IsNaN(float64(batchloss)) == true {
		panic("reach NAN")
	}
	return percent / float32(batchsize), batchloss / float32(batchsize)
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
