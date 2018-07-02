package main

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/pooling"
	"github.com/dereklstinson/GoCuNets/layers/softmax"
	"github.com/dereklstinson/GoCuNets/utils"
	"github.com/dereklstinson/GoCudnn"
)

func main() {

}

type section struct {
	hasfilter     bool
	filter        *cnn.Layer
	hasactivation bool
	activation    *activation.Layer
	haspooling    bool
	pooling       *pooling.Layer
	hassoftmax    bool
	softmax       *softmax.Layer
}

type layer struct {
	x    *layers.IO
	y    *layers.IO
	sect section
}

func createlayer(input *layers.IO, sect interface{}, output *layers.IO) (layer, error) {
	var l layer
	l.x = input
	l.y = output
	switch x := sect.(type) {
	case *cnn.Layer:
		l.sect.hasfilter = true
		l.sect.filter = x
	case *activation.Layer:
		l.sect.hasactivation = true
		l.sect.activation = x
	case *pooling.Layer:
		l.sect.haspooling = true
		l.sect.pooling = x
	case *softmax.Layer:
		l.sect.hassoftmax = true
		l.sect.softmax = x
	default:
		return l, errors.New("not supported sect")
	}
	return l, nil

}

type network struct {
	model []layer
}

func (net *network) Append(input layer, e error) {
	if e != nil {
		panic(e)
	}
	net.model = append(net.model, input)
}
func errcheck(err error) {
	if err != nil {
		panic(err)
	}
}
func model() network {
	handle := gocudnn.NewHandle()
	var Tensor gocudnn.Tensor
	shape := Tensor.Shape //simple arguments to array maker  x
	pad := shape
	dilation := shape
	stride := shape

	Float := Tensor.Flgs.Data.Float()
	NCHW := Tensor.Flgs.Format.NCHW()

	inputDesc, err := Tensor.NewTensor4dDescriptor(Float, NCHW, shape(1, 1, 28, 28))
	errcheck(err)
	size, err := inputDesc.GetSizeInBytes()
	errcheck(err)
	inputmemHost, err := makehostmem(size, Float, 0.0, false)
	errcheck(err)
	var Mem gocudnn.Memory
	HTD := Mem.Flgs.HostToDevice()
	inputmemDevice, err := gocudnn.Malloc(size)
	errcheck(err)
	err = gocudnn.CudaMemCopy(inputmemDevice, inputmemHost, size, HTD)
	errcheck(err)

	IO01 := layers.BuildIO(inputDesc, inputmemDevice, nil)
	var networkmod network

	/*
	   Layer 1 Convolution
	*/

	Layer1Helper := cnn.CreateLayerHelper()
	Conv1 := cnn.CreateConvolutionHelper()
	Tens1 := cnn.CreateTensorHelper()
	Layer1Helper.CoreSettings(false, Float, Conv1.Flgs.Mode.CrossCorrelation(), Tens1.Flgs.Format.NCHW())
	err = Layer1Helper.InputInsert(inputDesc)
	errcheck(err)
	err = Layer1Helper.FilterSetup(shape(20, 1, 5, 5), pad(1, 1), stride(1, 1), dilation(1, 1))
	errcheck(err)
	AlgoFwd1, AlgoBwdData1, AlgoBwdFilt1, _, err := Layer1Helper.GetBestAlgoConsidering(handle, 0, false)
	errcheck(err)
	inputDesc, Filter1D, ConvD1, Output1D := Layer1Helper.ReturnDescriptors()
	filter1size, err := Filter1D.TensorD().GetSizeInBytes()
	errcheck(err)
	convlayer1, err := cnn.LayerSetup(inputDesc, Filter1D, ConvD1, *AlgoFwd1, *AlgoBwdData1, *AlgoBwdFilt1, filter1size)
	errcheck(err)
	err = convlayer1.Build()

	errcheck(err)

	/*
		IO12
	*/

	outputsize1, err := Output1D.GetSizeInBytes()
	errcheck(err)
	outputmem12, err := gocudnn.Malloc(outputsize1)
	errcheck(err)
	doutputmem12, err := gocudnn.Malloc(outputsize1)
	errcheck(err)
	IO12 := layers.BuildIO(Output1D, outputmem12, doutputmem12)
	networkmod.Append(createlayer(IO01, convlayer1, IO12))
	/*
	   Layer 2 Activation
	*/
	var Activation gocudnn.Activation
	Activation1, err := Activation.NewActivationDescriptor(Activation.Flgs.Relu(), Tensor.Flgs.NaN.NotPropagateNan(), 0)
	errcheck(err)
	ActiveLayer2 := activation.LayerSetup(Activation1, gocudnn.CFloat(1), gocudnn.CFloat(0), gocudnn.CFloat(1), gocudnn.CFloat(0))

	/*
	   IO23
	*/

	outputmem23, err := gocudnn.Malloc(outputsize1)
	errcheck(err)
	doutputmem23, err := gocudnn.Malloc(outputsize1)
	errcheck(err)
	IO23 := layers.BuildIO(Output1D, outputmem23, doutputmem23)
	networkmod.Append(createlayer(IO12, ActiveLayer2, IO23))

	/*
		Layer 3 Pooling
	*/
	var Pool gocudnn.Pooling
	PoolD, err := Pool.NewPooling2dDescriptor(Pool.Flgs.Max(), Tensor.Flgs.NaN.NotPropagateNan(), shape(2, 2), shape(0, 0), shape(2, 2))
	errcheck(err)
	Layer3Pool := pooling.LayerSetup(PoolD, gocudnn.CFloat(1), gocudnn.CFloat(0), gocudnn.CFloat(1), gocudnn.CFloat(0))
	pooloutdims, err := PoolD.GetPoolingForwardOutputDim(Output1D)
	errcheck(err)

	/*
		IO34
	*/

	IO34TensorD, err := Tensor.NewTensor4dDescriptor(Float, NCHW, pooloutdims)
	errcheck(err)
	outputforIO34, err := IO34TensorD.GetSizeInBytes()
	errcheck(err)
	outputmem34, err := gocudnn.Malloc(outputforIO34)
	errcheck(err)
	doutputmem34, err := gocudnn.Malloc(outputforIO34)
	errcheck(err)
	IO34 := layers.BuildIO(IO34TensorD, outputmem34, doutputmem34)
	networkmod.Append(createlayer(IO23, Layer3Pool, IO34))
	/*
	   Layer 4 Convolution
	*/
	Layer4Helper := cnn.CreateLayerHelper()
	Conv4 := cnn.CreateConvolutionHelper()
	Tens4 := cnn.CreateTensorHelper()

	Layer4Helper.CoreSettings(false, Float, Conv4.Flgs.Mode.CrossCorrelation(), Tens4.Flgs.Format.NCHW())
	err = Layer4Helper.InputInsert(IO34TensorD)
	errcheck(err)
	err = Layer4Helper.FilterSetup(shape(20, 1, 5, 5), pad(1, 1), stride(1, 1), dilation(1, 1))
	errcheck(err)
	AlgoFwd4, AlgoBwdData4, AlgoBwdFilt4, _, err := Layer4Helper.GetBestAlgoConsidering(handle, 0, false)
	errcheck(err)
	inputDesc4, Filter4D, ConvD4, Output4D := Layer4Helper.ReturnDescriptors()
	filter4size, err := Filter4D.TensorD().GetSizeInBytes()
	errcheck(err)
	convlayer4, err := cnn.LayerSetup(inputDesc4, Filter4D, ConvD4, *AlgoFwd4, *AlgoBwdData4, *AlgoBwdFilt4, filter4size)
	errcheck(err)
	err = convlayer4.Build()

	errcheck(err)

	/*
		IO45
	*/

	outputsize4, err := Output4D.GetSizeInBytes()
	errcheck(err)
	outputmem45, err := gocudnn.Malloc(outputsize4)
	errcheck(err)
	doutputmem45, err := gocudnn.Malloc(outputsize4)
	errcheck(err)
	IO45 := layers.BuildIO(Output4D, outputmem45, doutputmem45)
	networkmod.Append(createlayer(IO34, convlayer4, IO45))
	/*
	   Layer 5 Activation
	*/

	Activation5, err := Activation.NewActivationDescriptor(Activation.Flgs.Relu(), Tensor.Flgs.NaN.NotPropagateNan(), 0)
	errcheck(err)
	alayer5 := activation.LayerSetup(Activation5, gocudnn.CFloat(1), gocudnn.CFloat(0), gocudnn.CFloat(1), gocudnn.CFloat(0))

	/*
	   IO56
	*/

	outputmem56, err := gocudnn.Malloc(outputsize4)
	errcheck(err)
	doutputmem56, err := gocudnn.Malloc(outputsize4)
	errcheck(err)
	IO56 := layers.BuildIO(Output4D, outputmem56, doutputmem56)

	networkmod.Append(createlayer(IO45, alayer5, IO56))

	/*
		Layer 6 Pooling
	*/

	PoolD6, err := Pool.NewPooling2dDescriptor(Pool.Flgs.Max(), Tensor.Flgs.NaN.NotPropagateNan(), shape(2, 2), shape(0, 0), shape(2, 2))
	errcheck(err)
	Layer6Pool := pooling.LayerSetup(PoolD6, gocudnn.CFloat(1), gocudnn.CFloat(0), gocudnn.CFloat(1), gocudnn.CFloat(0))
	pooloutdims6, err := PoolD.GetPoolingForwardOutputDim(Output4D)
	errcheck(err)

	/*
		IO67
	*/

	IO67TensorD, err := Tensor.NewTensor4dDescriptor(Float, NCHW, pooloutdims6)
	errcheck(err)
	outputforIO67, err := IO67TensorD.GetSizeInBytes()
	errcheck(err)
	outputmem67, err := gocudnn.Malloc(outputforIO67)
	errcheck(err)
	doutputmem67, err := gocudnn.Malloc(outputforIO67)
	errcheck(err)
	IO67 := layers.BuildIO(IO67TensorD, outputmem67, doutputmem67)
	networkmod.Append(createlayer(IO56, Layer6Pool, IO67))

	/*
		Layer 7 Convolution

	*/
	Layer7Helper := cnn.CreateLayerHelper()
	Conv7 := cnn.CreateConvolutionHelper()
	Tens7 := cnn.CreateTensorHelper()
	Layer7Helper.CoreSettings(false, Float, Conv7.Flgs.Mode.CrossCorrelation(), Tens7.Flgs.Format.NCHW())
	err = Layer7Helper.InputInsert(IO67TensorD)
	errcheck(err)
	pooloutdims6[0] = 10
	err = Layer7Helper.FilterSetup(pooloutdims6, pad(0, 0), stride(1, 1), dilation(1, 1))
	errcheck(err)
	AlgoFwd7, AlgoBwdData7, AlgoBwdFilt7, _, err := Layer4Helper.GetBestAlgoConsidering(handle, 0, false)
	errcheck(err)
	inputDesc7, Filter7D, ConvD7, Output7D := Layer7Helper.ReturnDescriptors()
	filter7size, err := Filter7D.TensorD().GetSizeInBytes()
	errcheck(err)
	convlayer7, err := cnn.LayerSetup(inputDesc7, Filter7D, ConvD7, *AlgoFwd7, *AlgoBwdData7, *AlgoBwdFilt7, filter7size)
	errcheck(err)
	err = convlayer7.Build()
	errcheck(err)
	/*
		IO78
	*/

	outputforIO78, err := Output7D.GetSizeInBytes()
	errcheck(err)
	outputmem78, err := gocudnn.Malloc(outputforIO78)
	errcheck(err)
	doutputmem78, err := gocudnn.Malloc(outputforIO78)
	errcheck(err)
	IO78 := layers.BuildIO(Output7D, outputmem78, doutputmem78)
	networkmod.Append(createlayer(IO67, convlayer7, IO78))

	/*
	 Layer 8	SoftMax
	*/

	softmax8 := softmax.BuildDefault()

	/*
		IO89
	*/

	outputforIO89, err := Output7D.GetSizeInBytes()
	errcheck(err)
	outputmem89, err := gocudnn.Malloc(outputforIO89)
	errcheck(err)
	doutputmem89, err := gocudnn.Malloc(outputforIO89)
	errcheck(err)
	IO89 := layers.BuildIO(Output7D, outputmem89, doutputmem89)
	networkmod.Append(createlayer(IO78, softmax8, IO89))

	return networkmod
}

func makehostmem(size gocudnn.SizeT, kind gocudnn.DataType, fanin float64, rand bool) (*gocudnn.GoPointer, error) {
	var flg gocudnn.DataTypeFlag
	switch kind {
	case flg.Float():
		x := make([]float32, size)
		if rand == true {
			for i := 0; i < int(size); i++ {
				x[i] = float32(utils.RandWeightSet(0, 1, fanin))
			}
		}

		return gocudnn.MakeGoPointer(x)

	default:
		return nil, errors.New("not supported")

	}
}
