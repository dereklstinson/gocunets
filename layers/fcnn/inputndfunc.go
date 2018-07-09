package fcnn

import (
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

func createNdfrominput(handle *gocudnn.Handle, neurons int32, input layers.IO) (*Layer, *layers.IO, error) {
	fmt, dtype, shape, err := input.Properties()
	if err != nil {
		return nil, nil, err
	}
	dims := len(shape)
	var convhelp gocudnn.Convolution
	f := convhelp.Flgs
	//var thelp gocudnn.Tensor

	x := shape[0] * shape[1]
	shape[0] = neurons
	shape[1] = x
	tens, err := layers.BuildIO(fmt, dtype, shape)

	if err != nil {
		tens.Destroy()
		return nil, nil, err
	}
	conv, err := convhelp.NewConvolutionNdDescriptor(f.Mode.CrossCorrelation(), dtype, padding(dims), convdescelse(dims), convdescelse(dims))
	if err != nil {
		tens.Destroy()
		conv.DestroyDescriptor()
		return nil, nil, err
	}
	outputdems, err := conv.GetConvolutionNdForwardOutputDim(input.T().TD(), tens.T().FD())
	if err != nil {
		return nil, nil, err
	}

	err = dimscheck(outputdems, maketest(neurons, dims))
	if err != nil {
		return nil, nil, err
	}
	output, err := layers.BuildIO(fmt, dtype, outputdems)
	if err != nil {
		return nil, nil, err
	}
	fwdalgo, err := convhelp.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle, input.T().TD(), tens.T().FD(), conv, output.T().TD(), convhelp.Flgs.Fwd.Pref.NoWorkSpace(), 0)
	if err != nil {
		return nil, nil, err
	}
	bwddata, err := convhelp.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle, tens.T().FD(), output.DeltaT().TD(), conv, input.T().TD(), convhelp.Flgs.Bwd.DataPref.NoWorkSpace(), 0)
	if err != nil {
		return nil, nil, err
	}
	bwdfilt, err := convhelp.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle, input.T().TD(), output.DeltaT().TD(), conv, tens.T().FD(), convhelp.Flgs.Bwd.FltrPref.NoWorkSpace(), 0)
	if err != nil {
		return nil, nil, err
	}
	bias, err := layers.BuildIO(fmt, dtype, outputdems)
	if err != nil {
		return nil, nil, err
	}
	return &Layer{neurons: tens, bias: bias, operation: conv, bwddata: bwddata, bwdfilt: bwdfilt, fwdalgo: fwdalgo}, output, nil
}
func padding(dims int) []int32 {
	return make([]int32, dims)
}
func convdescelse(dims int) []int32 {
	x := make([]int32, dims)
	for i := 0; i < dims; i++ {
		x[i] = int32(1)
	}
	return x
}
func maketest(neurons int32, dims int) []int32 {
	x := dims + 2
	array := make([]int32, x)
	array[0] = int32(neurons)
	for i := 1; i < len(array); i++ {
		array[i] = 1
	}
	return array
}
