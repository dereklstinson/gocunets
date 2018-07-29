package main

import (
	"fmt"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor/convolution"
	"github.com/dereklstinson/GoCuNets/testing/mnist/dfuncs"
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

	//input tensor
	input, err := layers.BuildIO(frmt, dtype, dims(1, 1, 28, 28), memmanaged)
	cherror(err)
	//Setting Up Network

	//Convolution Layer
	layer1, output1, err := cnn.AIOLayerSetupDefault(handle, input, filter(20, 1, 5, 5), CMode, padding(2, 2), stride(1, 1), dilation(1, 1), memmanaged)
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
	layer4, output4, err := fcnn.CreateFromInput(handle, int32(10), aoutput3, memmanaged)
	//Output Layer
	softmax, answer, err := softmax.BuildDefault(output4, memmanaged)
	cherror(err)
	//	actual, err := layers.BuildIO(fmt, dtype, []int32{1, 10, 1, 1}, true)
	//TrainingFunc
	batchsize := 16 // how many forward and backward runs before updating weights.

	//Setup Layer Trainers
	//Decay isn't available right now so................
	decay1, decay2 := 0.0, 0.0
	rate, momentum := .01, .6
	err = layer1.SetupTrainer(handle, decay1, decay2, rate, momentum)
	cherror(err)
	err = layer2.SetupTrainer(handle, decay1, decay2, rate, momentum)
	cherror(err)
	err = layer3.SetupTrainer(handle, decay1, decay2, rate, momentum)
	cherror(err)
	err = layer4.SetupTrainer(handle, decay1, decay2, rate, momentum)
	cherror(err)

	filedirectory := "/home/derek/go/src/github.com/dereklstinson/GoCuNets/testing/mnist/files/"
	trainingdata, err := dfuncs.LoadMNIST(filedirectory, "train-labels.idx1-ubyte", "train-images.idx3-ubyte")
	cherror(err)
	testingdata, err := dfuncs.LoadMNIST(filedirectory, "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte")
	cherror(err)
	trainepoch := len(trainingdata)
	fmt.Println("Length of Training Data", len(trainingdata))
	fmt.Println("Length of Testing Data", len(testingdata))
	//Since Data is so small we can load it all into the GPU

	for j := 0; j < trainepoch; {
		//training

		for i := 0; i < batchsize; i++ {
			j++
			//Will Need A Load Input Func
			//Forward Section
			err = layer1.ForwardProp(handle, nil, input, output1)
			cherror(err)
			err = activation1.ForwardProp(handle, output1, aoutput1)
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
			err = softmax.ForwardProp(handle, output4, answer)
			cherror(err)

			//Will Need an Actual answers func

			//Backward Section
			err = softmax.BackProp(handle, output4, answer)
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
			err = layer1.BackProp(handle, nil, input, output1)
			cherror(err)
		}
		err = layer1.UpdateWeights(handle)
		cherror(err)
		err = layer2.UpdateWeights(handle)
		cherror(err)
		err = layer3.UpdateWeights(handle)
		cherror(err)
		err = layer4.UpdateWeights(handle)
		cherror(err)
	}

	/*
		layer1, err := cnn.LayerSetup(fmt, dtype, dims(20, 1, 5, 5), CMode, dims(1, 1), dims(1, 1), dims(1, 1))
		cherror(err)
		inout1, err := layer1.MakeOutputTensor(input)
		cherror(err)
	*/

	//	c1.OutputDim(input)
	//	cnn.LayerSetup(cflgs.)
	err = devices[0].Reset()
	cherror(err)
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
