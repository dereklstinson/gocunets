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

	shape := gocudnn.Shape //simple arguments to array maker
	var flg gocudnn.Flags
	Float := flg.DataType.Float()
	NCHW := flg.TensorFormat.NCHW()

	inputDesc, err := gocudnn.NewTensor4dDescriptor(Float, NCHW, shape(1, 1, 28, 28))
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

	crosscorr := flg.ConvolutionMode.CrossCorrelation()
	convdesc, err := gocudnn.NewConvolution2dDescriptor(crosscorr, Float, shape(2, 2), shape(1, 1), shape(1, 1))
	ConvPref:=flg.ConvolutionFwdPref.NoWorkSpace()
	errcheck(err)
filter1,err:=	gocudnn.NewFilter4dDescriptor(Float,NCHW,shape(10,10,5,5)
errcheck(err)
    cnn.LayerSetup(inputDesc,filter1,convdesc,)
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
