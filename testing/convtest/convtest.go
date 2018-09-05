package main

import (
	"fmt"

	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCudnn"
)

func main() {
	handle := gocudnn.NewHandle()
	var cflags gocudnn.ConvolutionFlags
	var tflags gocudnn.TensorFlags
	frmt := tflags.Format.NHWC()
	dtype := tflags.Data.Float()
	cmode := cflags.Mode.CrossCorrelation()

	IO1, err := layers.BuildIO(frmt, dtype, slice(1, 5, 20, 20), true)
	if err != nil {
		panic(err)
	}
	pad := slice
	dilation := slice
	stride := slice
	filter := slice
	layer, IO2, err := cnn.AIOLayerSetupDefault(handle, IO1, filter(5, 5, 5, 5), cmode, pad(0, 0), stride(1, 1), dilation(1, 1), true)
	if err != nil {
		//		format, data, filtdims, err := layer.FilterProps()
		//		fmt.Println(format, data, filtdims)
		panic(err)
	}
	format, data, filtdims, err := layer.FilterProps()
	if err != nil {
		panic(err)
	}
	fmt.Println(format, data, filtdims)

	_, _, dims, err := IO2.Properties()
	if err != nil {
		panic(err)
	}
	fmt.Println(dims)
}

func slice(args ...int32) []int32 {
	return args
}
