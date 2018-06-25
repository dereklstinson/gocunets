package main

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/utils"
	"github.com/dereklstinson/GoCudnn"
)

func main() {

}

type section struct {
	inout  bool
	io     *layers.IO
	filter bool
	filt   *cnn.Layer
}
type network struct {
	model []section
}

func (net *network) Append(input interface{}) {
	switch x := input.(type) {
	case *cnn.Layer:
		var sect section
		sect.filter = true
		sect.filt = x
		net.model = append(net.model, sect)
	case *layers.IO:
		var sect section
		sect.inout = true
		sect.io = x
		net.model = append(net.model, sect)
	}
}
func errcheck(err error) {
	if err != nil {
		panic(err)
	}
}
func model() network {
	handle := gocudnn.NewHandle()
	shape := gocudnn.Shape //simple arguments to array maker  x
	var flg gocudnn.Flags
	Float := flg.DataType.Float()
	NCHW := flg.TensorFormat.NCHW()
	var Tensor gocudnn.Tensor

	inputDesc, err := Tensor.NewTensor4dDescriptor(Float, NCHW, shape(1, 1, 28, 28))
	errcheck(err)
	size, err := inputDesc.GetSizeInBytes()
	errcheck(err)
	inputmemHost, err := makehostmem(size, Float, 0.0, false)
	errcheck(err)
	HTD := flg.MemcpyKind.HostToDevice()
	inputmemDevice, err := gocudnn.Malloc(size)
	errcheck(err)
	err = gocudnn.CudaMemCopy(inputmemDevice, inputmemHost, size, HTD)
	errcheck(err)

	inputlayer := layers.BuildIO(inputDesc, inputmemDevice, nil)
	var networkmod network

	networkmod.Append(inputlayer)
	/*
	   Layer 1
	*/
	var Convolution gocudnn.Convolution
	crosscorr := flg.ConvolutionMode.CrossCorrelation()
	ConvD1, err := Convolution.NewConvolution2dDescriptor(crosscorr, Float, shape(2, 2), shape(1, 1), shape(1, 1))

	errcheck(err)
	var Filter gocudnn.Filter
	filter1, err := Filter.NewFilter4dDescriptor(Float, NCHW, shape(10, 1, 5, 5))
	errcheck(err)
	outputdims, err := ConvD1.GetConvolution2dForwardOutputDim(inputDesc, filter1)
	errcheck(err)
	outputdesc1, err := Tensor.NewTensor4dDescriptor(Float, NCHW, outputdims)
	errcheck(err)
	ConvFWDAlgo1, err := Convolution.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle, inputDesc, filter1, ConvD1, outputdesc1, Convolution.Flgs.Fwd.Pref.NoWorkSpace(), 0)
	errcheck(err)
	ConvBWdAlgoData1, err := Convolution.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle, filter1, outputdesc1, ConvD1, inputDesc, Convolution.Flgs.Bwd.DataPref.NoWorkSpace(), 0)
	errcheck(err)
	ConvBWdAlgoFilt1, err := Convolution.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle, inputDesc, outputdesc1, ConvD1, filter1, Convolution.Flgs.Bwd.FltrPref.NoWorkspace(), 0)
	errcheck(err)
	filtsize, err := filter1.TensorD().GetSizeInBytes()
	errcheck(err)
	layer1, err := cnn.LayerSetup(inputDesc, filter1, ConvD1, ConvFWDAlgo1, ConvBWdAlgoData1, ConvBWdAlgoFilt1, filtsize)
	errcheck(err)
	networkmod.Append(layer1)
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
