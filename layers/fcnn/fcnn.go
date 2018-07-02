package fcnn

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is a fcnn layer for a network
type Layer struct {
	neurons     *tensor.Tensor
	operation   *gocudnn.ConvolutionD
	convfwdalgo *gocudnn.ConvFwdAlgo
	convbwddata *gocudnn.ConvBwdDataAlgo
	convbwdfilt *gocudnn.ConvBwdFiltAlgo
}

func CreateFromInput(handle *gocudnn.Handle, neurons int32, input layers.IO) (*Layer, *layers.IO, error) {
	fmt, dtype, shape, err := some.Properties()
	if err != nil {
		return nil, nil, err
	}
	var convhelp gocudnn.Convolution
	f := convhelp.Flgs
	var thelp gocudnn.Tensor
	sh := thelp.Shape
	if len(shape) == 4 {
		x := shape[0] * shape[1]
		shape[0] = neurons
		shape[1] = x
		tens, err := tensor.Create(fmt, dtype, shape)

		if err != nil {
			tens.Destroy()
			return nil, nil, err
		}
		conv, err := convhelp.NewConvolution2dDescriptor(f.Mode.CrossCorrelation(), dtype, sh(0, 0), sh(1, 1), sh(1, 1))
		if err != nil {
			tens.Destroy()
			conv.DestroyDescriptor()
			return nil, nil, err
		}
		outputdems, err := conv.GetConvolution2dForwardOutputDim(input.Tensor(), tens.FilterD())
		if err != nil {
			return nil, nil, err
		}
		err = dimscheck(outputdems, []int32{neurons, 1, 1, 1})
		if err != nil {
			return nil, nil, err
		}
		output, err := tensor.Create(fmt, dtype, outputdems)
		if err != nil {
			return nil, nil, err
		}
		fwdalgo, err := convhelp.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle, some.Tensor(), tens.FilterD, conv, output.TensorD, convhelp.Flgs.Fwd.Pref.NoWorkSpace, 0)
		if err != nil {
			return nil, nil, err
		}
		return nil, &Layer{neurons: tens, operation: conv}, nil
	}
	return nil, nil, nil

}

func dimscheck(a, b []int32) error {
	if len(a) != len(b) {
		return errors.New("num of dims not same")
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return fmt.Errorf("a[%d]=%d,b[%d]=%d", i, a[i], i, b[i])
		}
	}
	return nil

}
