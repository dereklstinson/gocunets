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

func model() network {

	shape := gocudnn.Shape
	var flg gocudnn.Flags
	Float := flg.DataType.Float()
	NCHW := flg.TensorFormat.NCHW()

	inputDesc, err := gocudnn.NewTensor4dDescriptor(Float, NCHW, shape(1, 1, 28, 28))
	if err != nil {
		panic(err)

	}
	size, err := inputDesc.GetSizeInBytes()
	if err != nil {
		panic(err)
	}
	inputmemHost, err := MakeHostMem(size, Float, 0.0, false)
	if err != nil {
		panic(err)
	}
	HTD := flg.MemcpyKind.HostToDevice()
	inputmemDevice, err := gocudnn.Malloc(size)
	if err != nil {
		panic(err)
	}
	err = gocudnn.CudaMemCopy(inputmemDevice, inputmemHost, size, HTD)
	if err != nil {
		panic(err)
	}

	inputlayer := layers.BuildIO(inputDesc, inputmemDevice, nil)
	var networkmod network
	networkmod.Append(inputlayer)
	return networkmod
}

func MakeHostMem(size gocudnn.SizeT, kind gocudnn.DataType, fanin float64, rand bool) (*gocudnn.GoPointer, error) {
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
