package main

import (
	"fmt"
	//"math"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor/convolution"
	//	"github.com/dereklstinson/GoCuNets/testing/mnist/dfuncs"
	"github.com/dereklstinson/GoCudnn"
	//	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	//	"github.com/dereklstinson/GoCuNets/layers/fcnn"
	"github.com/dereklstinson/GoCuNets/layers/pooling"
	//"github.com/dereklstinson/GoCuNets/layers/softmax"
	//	gocudnn "github.com/dereklstinson/GoCudnn"
)

func main() {
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
	stream, err := gocudnn.CreateBlockingStream()
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
	NanProp := gocudnn.PropagationNANFlag{}.PropagateNan()
	memmanaged := true
	//	dims := gocudnn.Tensor.Shape
	filter := dims
	padding := dims
	stride := dims
	dilation := dims

	//input tensor
	input, err := layers.BuildIO(frmt, dtype, dims(1, 1, 6, 6), memmanaged)
	cherror(err)
	inputvalues := []float32{-1, 0, 1, 2, 3, 4,
		1, 2, 3, 4, 5, 6,
		-3, -2, -1, 0, 1, 2,
		0, 1, 2, 3, 4, 5,
		-2, -1, 0, 1, 2, 3,
		2, 3, 4, 5, 6, 7,
	}
	inptr, err := gocudnn.MakeGoPointer(inputvalues)
	cherror(err)
	err = input.LoadTValues(inptr)
	cherror(err)
	//answervalues := []float32{0, 0, 1}
	//ansptr, err := gocudnn.MakeGoPointer(answervalues)
	//cherror(err)
	//err = input.LoadDeltaTValues(ansptr)
	//cherror(err)

	//Setting Up Network

	//Convolution Layer
	layer1, output1, err := cnn.AIOLayerSetupDefault(handle, input, filter(3, 1, 5, 5), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged)
	cherror(err)
	layer1weights := []float32{1, 0, 1, 0, 1,
		0, 1, 0, 1, 0,
		1, 0, 1, 0, 1,
		0, 1, 0, 1, 0,
		1, 0, 1, 0, 1,

		0, 2, 0, 2, 0,
		2, 0, 2, 0, 2,
		0, 2, 0, 2, 0,
		2, 0, 2, 0, 2,
		0, 2, 0, 2, 0,

		-1, 0, -1, 0, -1,
		0, -1, 0, -1, 0,
		-1, 0, -1, 0, -1,
		0, -1, 0, -1, 0,
		-1, 0, -1, 0, -1,
	}
	err = layer1.LoadWValues(layer1weights)
	cherror(err)
	//Math Note: output= ((input-filter+2*padding)/stride) +1 -> (6-5/1)  +1 =2

	//Activation Layer
	activation1, aoutput1, err := activation.LayerSetup(output1, AMode, NanProp, 10.0, 1.0, 0.0, 1.0, 0.0, memmanaged)
	cherror(err)
	//pooling layer
	pooling1, poutput1, err := pooling.LayerSetup(Pmode, NanProp, aoutput1, filter(2, 2), padding(0, 0), stride(1, 1), memmanaged)
	cherror(err)
	err = layer1.ForwardProp(handle, nil, input, output1)
	cherror(err)
	_, _, odims, err := output1.Properties()
	cherror(err)
	output1slice := make([]float32, odims[0]*odims[1]*odims[2]*odims[3])
	output1.T().Memer().FillSlice(output1slice)
	fmt.Println("Layer1 Output: ", output1slice)
	err = activation1.ForwardProp(handle, output1, aoutput1)
	cherror(err)
	aoutput1slice := make([]float32, odims[0]*odims[1]*odims[2]*odims[3])
	err = aoutput1.T().Memer().FillSlice(aoutput1slice)
	cherror(err)
	fmt.Println("Activation Output: ", aoutput1slice)

	err = pooling1.ForwardProp(handle, aoutput1, poutput1)
	poutput1slice := make([]float32, 3)
	err = poutput1.T().Memer().FillSlice(poutput1slice)
	cherror(err)
	fmt.Println("Pooling Slice: ", poutput1slice)

}

func cherror(input error) {
	if input != nil {
		fmt.Println("***************************")
		panic(input)

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
