package fcnn

import (
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

func create4dfrominput(handle *gocudnn.Handle, neurons int32, input layers.IO) (*Layer, *layers.IO, error) {
	fmt, dtype, shape, err := input.Properties()
	if err != nil {
		return nil, nil, err
	}
	var convhelp gocudnn.Convolution
	f := convhelp.Flgs
	var thelp gocudnn.Tensor
	sh := thelp.Shape

	x := shape[0] * shape[1]
	shape[0] = neurons
	shape[1] = x
	tens, err := layers.BuildIO(fmt, dtype, shape)

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
	outputdems, err := conv.GetConvolution2dForwardOutputDim(input.Tensor().TD(), tens.Tensor().FD())
	if err != nil {
		return nil, nil, err
	}
	err = dimscheck(outputdems, []int32{neurons, 1, 1, 1})
	if err != nil {
		return nil, nil, err
	}
	output, err := layers.BuildIO(fmt, dtype, outputdems)
	if err != nil {
		return nil, nil, err
	}
	fwdalgo, err := convhelp.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle, input.Tensor().TD(), tens.Tensor().FD(), conv, output.Tensor().TD(), convhelp.Flgs.Fwd.Pref.NoWorkSpace(), 0)
	if err != nil {
		return nil, nil, err
	}
	bwddata, err := convhelp.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle, tens.Tensor().FD(), output.DTensor().TD(), conv, input.Tensor().TD(), convhelp.Flgs.Bwd.DataPref.NoWorkSpace(), 0)
	if err != nil {
		return nil, nil, err
	}
	bwdfilt, err := convhelp.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle, input.Tensor().TD(), output.DTensor().TD(), conv, tens.Tensor().FD(), convhelp.Flgs.Bwd.FltrPref.NoWorkSpace(), 0)
	if err != nil {
		return nil, nil, err
	}
	bias, err := layers.BuildIO(fmt, dtype, sh(neurons, 1, 1, 1))
	if err != nil {
		return nil, nil, err
	}
	return &Layer{neurons: tens, bias: bias, operation: conv, bwddata: bwddata, bwdfilt: bwdfilt, fwdalgo: fwdalgo}, output, nil
}
