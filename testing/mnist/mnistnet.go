package main

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor/convolution"
	"github.com/dereklstinson/GoCudnn"
	//	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/fcnn"
	"github.com/dereklstinson/GoCuNets/layers/pooling"
	"github.com/dereklstinson/GoCuNets/layers/softmax"
)

func main() {
	//cudnn context
	handle := gocudnn.NewHandle()
	var dtypeflags gocudnn.DataTypeFlag
	var fmtflags gocudnn.TensorFormatFlag
	//flags to set up network

	fmt := fmtflags.NCHW()
	dtype := dtypeflags.Float()
	CMode := convolution.Flags().Mode.CrossCorrelation() //.CrossCorrelation()
	AMode := gocudnn.ActivationModeFlag{}.Relu()
	NanProp := gocudnn.PropagationNANFlag{}.NotPropagateNan()
	//	dims := gocudnn.Tensor.Shape

	//input tensor
	input, err := layers.BuildIO(fmt, dtype, dims(1, 1, 28, 28))
	cherror(err)
	//Setting Up Network
	//Convolution Layer
	layer1, output1, err := cnn.AIOLayerSetupDefault(handle, input, dims(20, 1, 5, 5), CMode, dims(1, 1), dims(1, 1), dims(1, 1))
	cherror(err)
	//Activation Layer
	activation1, aoutput1, err := activation.LayerSetup(output1, Amode, NanProp, 10.0, 1.0, 0.0, 1.0, 0.0)
	cherror(err)
	pooling1, err := pooling.LayerSetup()
	//Convolution Layer
	layer2, output2, err := cnn.AIOLayerSetupDefault(handle, aoutput1, dims(20, 1, 5, 5), CMode, dims(1, 1), dims(1, 1), dims(1, 1))
	cherror(err)
	//Activation Layer
	activation2, aoutput2, err := activation.LayerSetup(output2, Amode, NanProp, 10.0, 1.0, 0.0, 1.0, 0.0)
	cherror(err)
	//Convolution Layer
	layer3, output3, err := cnn.AIOLayerSetupDefault(handle, aoutput2, dims(20, 1, 5, 5), CMode, dims(1, 1), dims(1, 1), dims(1, 1))
	cherror(err)
	//Activation Layer
	activation3, aoutput3, err := activation.LayerSetup(output3, Amode, NanProp, 10.0, 1.0, 0.0, 1.0, 0.0)
	cherror(err)
	//Fully Connected Layer ////Modified Convolution Layer :-)
	layer4, output4, err := fcnn.CreateFromInput(handle, int32(10), aoutput3)
	//Output Layer
	softmax, answer, err := softmax.BuildDefault(output4)
	cherror(err)

	//TrainingFunc
	batchsize := 16 // how many forward and backward runs before updating weights.

	for {
		//training
		for i := 0; i < len(batchsize); i++ {
			//Will Need A Load Input Func
			//Forward Section
			err = layer1.ForwardProp(handle, nil, input, output1)
			cherror(err)
			err = activation1.ForwardProp(handle, output1, aoutput1)
			cherror(err)
			err = layer2.ForwardProp(handle, nil, aoutput1, output2)
			cherror(err)
			err = activation2.ForwardProp(handle, output2, aoutput2)
			cherror(err)
			err = layer3.ForwardProp(handle, nil, aoutput2, output3)
			cherror(err)
			err = activation3.ForwardProp(handle, output3, aoutput3)
			cherror(err)
			err = layer4.ForwardProp()
			err = softmax.ForwardProp(handle, Output4, answer)
		}

	}

	/*
		layer1, err := cnn.LayerSetup(fmt, dtype, dims(20, 1, 5, 5), CMode, dims(1, 1), dims(1, 1), dims(1, 1))
		cherror(err)
		inout1, err := layer1.MakeOutputTensor(input)
		cherror(err)
	*/

	//	c1.OutputDim(input)
	//	cnn.LayerSetup(cflgs.)
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
		panic(input)
	}
}
