package convolution

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//AlgoLists Algo lists returns slices of performances for the fwd algos and bwd algos
func (c *Ops) AlgoLists(handle *cudnn.Handler, x, dx, w, dw, y, dy *tensor.Volume) ([]gocudnn.ConvFwdAlgoPerformance, []gocudnn.ConvBwdDataAlgoPerformance, []gocudnn.ConvBwdFiltAlgoPerformance, error) {
	maxfwd, err := c.helper.Funcs.Fwd.GetConvolutionForwardAlgorithmMaxCount(handle.Cudnn())
	if err != nil {
		return nil, nil, nil, err
	}
	fwdlist, err := c.helper.Funcs.Fwd.FindConvolutionForwardAlgorithm(handle.Cudnn(), x.TD(), w.FD(), c.desc, y.TD(), maxfwd)
	if err != nil {
		return nil, nil, nil, err
	}
	maxbwddata, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataAlgorithmMaxCount(handle.Cudnn())
	if err != nil {
		return nil, nil, nil, err
	}
	bwddata, err := c.helper.Funcs.Bwd.FindConvolutionBackwardDataAlgorithm(handle.Cudnn(), w.FD(), dy.TD(), c.desc, dx.TD(), maxbwddata)
	if err != nil {
		return nil, nil, nil, err
	}
	maxbwdfilt, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithmMaxCount(handle.Cudnn())
	if err != nil {
		return nil, nil, nil, err
	}
	bwdfilt, err := c.helper.Funcs.Bwd.FindConvolutionBackwardFilterAlgorithm(handle.Cudnn(), x.TD(), dy.TD(), c.desc, dw.FD(), maxbwdfilt)
	if bwdfilt != nil {
		return nil, nil, nil, err
	}
	return fwdlist, bwddata, bwdfilt, nil
}

//OutputDim will return the dims of what the output tensor should be
func (c *Ops) OutputDim(input *tensor.Volume, filter *tensor.Volume) ([]int32, error) {
	_, _, dims, err := input.Properties()
	if err != nil {
		return nil, err
	}
	_, _, fdims, err := filter.Properties()

	if err != nil {
		return nil, err
	}

	if len(dims) != len(fdims) {
		return nil, errors.New("length of dims not same")
	}
	if len(dims) == 4 {
		return c.desc.GetConvolution2dForwardOutputDim(input.TD(), filter.FD())
	}
	return c.desc.GetConvolutionNdForwardOutputDim(input.TD(), filter.FD())

}
