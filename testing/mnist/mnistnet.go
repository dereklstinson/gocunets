package main

import (
	"fmt"
	"math"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor/convolution"
	"github.com/dereklstinson/GoCuNets/testing/mnist/dfuncs"
	"github.com/dereklstinson/GoCuNets/trainer"
	"github.com/dereklstinson/GoCudnn"
	//	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/fcnn"
	"github.com/dereklstinson/GoCuNets/layers/pooling"
	"github.com/dereklstinson/GoCuNets/layers/softmax"
	//	gocudnn "github.com/dereklstinson/GoCudnn"
)

func main() {

	trainingkernellocation := "/home/derek/go/src/github.com/dereklstinson/GoCudnn/kernels/"
	gocudnn.Cuda{}.LockHostThread()
	//cudnn context
	var cuda gocudnn.Cuda
	//cuda.
	devices, err := cuda.GetDeviceList()
	cherror(err)
	devicenum := len(devices)
	fmt.Println("Number of Devices:", devicenum)
	err = devices[0].Set()
	cherror(err)
	handle := gocudnn.NewHandle()
	stream, err := gocudnn.Cuda{}.CreateBlockingStream()
	cherror(err)
	err = handle.SetStream(stream)
	cherror(err)
	var dtypeflags gocudnn.DataTypeFlag
	var fmtflags gocudnn.TensorFormatFlag
	//flags to set up network

	frmt := fmtflags.NCHW()
	dtype := dtypeflags.Float()
	CMode := convolution.Flags().Mode.CrossCorrelation() //.CrossCorrelation()
	AMode := gocudnn.ActivationModeFlag{}.Relu()
	Pmode := gocudnn.PoolingModeFlag{}.Max()
	NanProp := gocudnn.PropagationNANFlag{}.NotPropagateNan()
	memmanaged := true
	//	dims := gocudnn.Tensor.Shape
	filter := dims
	padding := dims
	stride := dims
	dilation := dims

	/*
	   Lets go ahead and start loading the training data

	*/

	batchsize := 20 // how many forward and backward runs before updating weights.

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

	batchnum := 0 //Im lazy so as I am making the batched data I am going to count it

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
		batchnum++
	}
	fmt.Println("Done Loading Training to GPU")

	testbatchnum := 0

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
		testbatchnum++
	}

	fmt.Println("Done Loading Testing To GPU")

	cherror(err)

	//Setting Up Network

	//Convolution Layer
	layer1, output1, err := cnn.AIOLayerSetupDefault(handle, gputrainingdata[0], filter(20, 1, 5, 5), CMode, padding(2, 2), stride(1, 1), dilation(1, 1), memmanaged)
	cherror(err)
	//Math Note: output= ((input-filter+2*padding)/stride) +1 -> (28-5+4/1)  +1 =28

	//Activation Layer
	activation1, aoutput1, err := activation.LayerSetup(output1, AMode, NanProp, 10.0, 1.0, 0.0, 1.0, 0.0, memmanaged)
	cherror(err)
	//pooling layer
	pooling1, poutput1, err := pooling.LayerSetup(Pmode, NanProp, aoutput1, filter(2, 2), padding(0, 0), stride(2, 2), memmanaged)
	cherror(err)
	//MathNote: output= ((input-filter+2*padding)/stride) +1 -> 28-2/2 +1 = 14

	//Convolution Layer
	layer2, output2, err := cnn.AIOLayerSetupDefault(handle, poutput1, filter(20, 20, 5, 5), CMode, padding(2, 2), stride(1, 1), dilation(1, 1), memmanaged)
	cherror(err)
	//MathNote: output= ((input-filter+2*padding)/stride) +1 -> (14-5+4/1) +1 =14

	//Activation Layer
	activation2, aoutput2, err := activation.LayerSetup(output2, AMode, NanProp, 10.0, 1.0, 0.0, 1.0, 0.0, memmanaged)
	cherror(err)
	//pooling layer
	pooling2, poutput2, err := pooling.LayerSetup(Pmode, NanProp, aoutput2, filter(2, 2), padding(0, 0), stride(2, 2), memmanaged)
	cherror(err)
	//MathNote: output= ((input-filter+2*padding)/stride) +1 -> (14-2/2) +1 =7

	//Convolution Layer
	layer3, output3, err := cnn.AIOLayerSetupDefault(handle, poutput2, filter(20, 20, 3, 3), CMode, padding(1, 1), stride(2, 2), dilation(1, 1), memmanaged)
	cherror(err)
	//MathNote: output= ((input-filter+2*padding)/stride) +1 -> (7-3+2/2) +1 =4

	//Activation Layer
	activation3, aoutput3, err := activation.LayerSetup(output3, AMode, NanProp, 10.0, 1.0, 0.0, 1.0, 0.0, memmanaged)
	cherror(err)
	//pooling layer
	pooling3, poutput3, err := pooling.LayerSetup(Pmode, NanProp, aoutput3, filter(2, 2), padding(0, 0), stride(2, 2), memmanaged)
	cherror(err)
	//MathNote: output= ((input-filter+2*padding)/stride) +1 -> (4-2/2) +1 =2

	//Fully Connected Layer ////Modified Convolution Layer :-)
	layer4, output4, err := fcnn.CreateFromInput(handle, int32(10), poutput3, memmanaged)
	//Output Layer
	softmax, err := softmax.BuildDefault(output4, gpuanswersdata[0])
	cherror(err)
	//	actual, err := layers.BuildIO(fmt, dtype, []int32{1, 10, 1, 1}, true)
	//TrainingFunc

	//Setup Layer Trainers
	//Decay isn't available right now so................
	decay1, decay2 := float32(0.00001), float32(0.0001)

	tctx, err := trainer.CreateAdamContext(0, devices[0], trainingkernellocation)

	cherror(err)
	l1trainer, err := trainer.SetupAdam(tctx, decay1, decay2, batchsize)
	cherror(err)
	l1btrainer, err := trainer.SetupAdam(tctx, decay1, decay2, batchsize)
	cherror(err)
	err = layer1.LoadTrainer(tctx, l1trainer, l1btrainer)
	cherror(err)

	l2trainer, err := trainer.SetupAdam(tctx, decay1, decay2, batchsize)
	cherror(err)
	l2btrainer, err := trainer.SetupAdam(tctx, decay1, decay2, batchsize)
	cherror(err)
	err = layer2.LoadTrainer(tctx, l2trainer, l2btrainer)
	cherror(err)

	l3trainer, err := trainer.SetupAdam(tctx, decay1, decay2, batchsize)
	cherror(err)
	l3btrainer, err := trainer.SetupAdam(tctx, decay1, decay2, batchsize)
	cherror(err)
	err = layer3.LoadTrainer(tctx, l3trainer, l3btrainer)
	cherror(err)

	l4trainer, err := trainer.SetupAdam(tctx, decay1, decay2, batchsize)
	cherror(err)
	l4btrainer, err := trainer.SetupAdam(tctx, decay1, decay2, batchsize)
	cherror(err)
	err = layer4.LoadTrainer(tctx, l4trainer, l4btrainer)
	cherror(err)
	/*
		err = layer1.SetupTrainer(handle, decay1, decay2, rate, momentum)
		cherror(err)
		err = layer2.SetupTrainer(handle, decay1, decay2, rate, momentum)
		cherror(err)
		err = layer3.SetupTrainer(handle, decay1, decay2, rate, momentum)
		cherror(err)
		err = layer4.SetupTrainer(handle, decay1, decay2, rate, momentum)
		cherror(err)
	*/
	epochs := 20
	//	inputslicefromgpumem := make([]float32, 28*28)
	for k := 0; k < epochs; k++ {

		for j := 0; j < batchnum; j++ { //I add the j++ at the end of this

			err = layer1.ForwardProp(handle, nil, gputrainingdata[j], output1)
			cherror(err)

			err = activation1.ForwardProp(handle, output1, aoutput1)

			cherror(err)

			cherror(err)
			err = pooling1.ForwardProp(handle, aoutput1, poutput1)
			cherror(err)
			err = layer2.ForwardProp(handle, nil, poutput1, output2)
			cherror(err)
			err = activation2.ForwardProp(handle, output2, aoutput2)
			cherror(err)
			err = pooling2.ForwardProp(handle, aoutput2, poutput2)
			cherror(err)
			err = layer3.ForwardProp(handle, nil, poutput2, output3)
			cherror(err)
			err = activation3.ForwardProp(handle, output3, aoutput3)
			cherror(err)
			err = pooling3.ForwardProp(handle, aoutput3, poutput3)
			cherror(err)
			err = layer4.ForwardProp(handle, poutput3, output4)
			cherror(err)
			err = softmax.ForwardProp(handle, output4, gpuanswersdata[j])
			cherror(err)
			err = stream.Sync()
			cherror(err)
			//Will Need an Actual answers func

			cherror(err)
			//Backward Section
			//	netoutput := make([]float32, 10*batchsize)
			//	desiredoutput := make([]float32, 10*batchsize)
			//	err = gpuanswersdata[j].T().Memer().FillSlice(netoutput)
			//	cherror(err)
			//	err = gpuanswersdata[j].DeltaT().Memer().FillSlice(desiredoutput)
			//	cherror(err)

			err = softmax.BackProp(handle, output4, gpuanswersdata[j])

			cherror(err)
			err = layer4.BackProp(handle, poutput3, output4)
			cherror(err)

			err = pooling3.BackProp(handle, aoutput3, poutput3)
			cherror(err)
			err = activation3.BackProp(handle, output3, aoutput3)
			cherror(err)
			err = layer3.BackProp(handle, nil, poutput2, output3)
			cherror(err)
			err = pooling2.BackProp(handle, aoutput2, poutput2)
			cherror(err)
			err = activation2.BackProp(handle, output2, aoutput2)
			cherror(err)
			err = layer2.BackProp(handle, nil, poutput1, output2)
			cherror(err)
			err = pooling1.BackProp(handle, aoutput1, poutput1)
			cherror(err)
			err = activation1.BackProp(handle, output1, aoutput1)
			cherror(err)
			err = layer1.BackProp(handle, nil, gputrainingdata[j], output1)
			cherror(err)

			cherror(err)
			/*
				go func(netoutput []float32, desiredoutput []float32, k int, batchsize int) {
					percent, loss := batchoutputchecker(netoutput, desiredoutput, batchsize, 10)
					fmt.Println("Epoch Percent Correct: ", percent, "		Epoch Loss: ", loss, "                  Epoch Number: ", k)

				}(netoutput, desiredoutput, k, batchsize)
			*/
			//fmt.Println(netoutput)
			//fmt.Println(desiredoutput)
			err = layer1.UpdateWeights(tctx)
			cherror(err)
			err = layer2.UpdateWeights(tctx)
			cherror(err)
			err = layer3.UpdateWeights(tctx)
			cherror(err)
			err = layer4.UpdateWeights(tctx)
			cherror(err)

		}
		netoutput := make([][]float32, testbatchnum)
		desiredoutput := make([][]float32, testbatchnum)
		for j := 0; j < testbatchnum; j++ {
			err = layer1.ForwardProp(handle, nil, gputestingdata[j], output1)
			cherror(err)

			err = activation1.ForwardProp(handle, output1, aoutput1)

			cherror(err)

			cherror(err)
			err = pooling1.ForwardProp(handle, aoutput1, poutput1)
			cherror(err)
			err = layer2.ForwardProp(handle, nil, poutput1, output2)
			cherror(err)
			err = activation2.ForwardProp(handle, output2, aoutput2)
			cherror(err)
			err = pooling2.ForwardProp(handle, aoutput2, poutput2)
			cherror(err)
			err = layer3.ForwardProp(handle, nil, poutput2, output3)
			cherror(err)
			err = activation3.ForwardProp(handle, output3, aoutput3)
			cherror(err)
			err = pooling3.ForwardProp(handle, aoutput3, poutput3)
			cherror(err)
			err = layer4.ForwardProp(handle, poutput3, output4)
			cherror(err)
			err = softmax.ForwardProp(handle, output4, gputestansdata[j])
			cherror(err)
			err = stream.Sync()
			cherror(err)
			//Will Need an Actual answers func

			cherror(err)
			//Backward Section
			netoutput[j] = make([]float32, 10*batchsize)
			desiredoutput[j] = make([]float32, 10*batchsize)
			err = gputestansdata[j].T().Memer().FillSlice(netoutput[j])
			cherror(err)
			err = gputestansdata[j].DeltaT().Memer().FillSlice(desiredoutput[j])
			cherror(err)
		}

		go func(netoutput [][]float32, desiredoutput [][]float32, k int, testbatchnum int, batchsize int) {
			percent, loss := epocoutputchecker(netoutput, desiredoutput, testbatchnum, batchsize, 10)
			fmt.Println("Epoch Percent Correct: ", percent, "		Epoch Loss: ", loss, "                  Epoch Number: ", k)

		}(netoutput, desiredoutput, k, testbatchnum, batchsize)

	}

	gocudnn.Cuda{}.UnLockHostThread()
	err = devices[0].Reset()
	cherror(err)
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
func InputData(data []dfuncs.LabeledData, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32, managed bool) ([]*layers.IO, error) {
	inputs := make([]*layers.IO, len(data))
	var err error
	for i := 0; i < len(inputs); i++ {
		inputs[i], err = layers.BuildIO(frmt, dtype, dims, managed)
		if err != nil {
			return nil, err
		}
		dataptr, err := gocudnn.MakeGoPointer(data[i].Data)
		if err != nil {
			return nil, err
		}
		err = inputs[i].LoadTValues(dataptr)
		if err != nil {
			return nil, err
		}
	}

	return inputs, nil
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
