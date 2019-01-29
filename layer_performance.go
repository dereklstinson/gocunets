package gocunets

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/Nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/Nvidia/cudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers"
)

const debuglayerperformance = false

func (l *layer) setcudnnperformancefwd(handle *cudnn.Handler, fwd convolution.ForwardPerformance) {
	if l.cnn != nil {
		l.cnn.SetFwdAlgoPerformance(handle, fwd)
		return
	}
	if l.cnntranspose != nil {
		//Since cnntranspose runs backwards the forward prop is actually the backprop data algo.
		l.cnntranspose.SetBwdDataAlgoPerformance(handle, fwd)
		return
	}
}
func (l *layer) setcudnnperformancebwdd(handle *cudnn.Handler, bwdd convolution.BackDataPerformance) {
	if l.cnn != nil {
		l.cnn.SetBwdDataAlgoPerformance(handle, bwdd)
		return
	}
	if l.cnntranspose != nil {
		//Since cnntranspose runs backwards the backprop for the data is actually the forward propagation algo
		l.cnntranspose.SetFwdAlgoPerformance(handle, bwdd)
		return
	}
}
func (l *layer) setcudnnperformancebwdf(handle *cudnn.Handler, bwdf convolution.BackFilterPerformance) {
	if l.cnn != nil {
		l.cnn.SetBwdFiltAlgoPerformance(handle, bwdf)
		return
	}
	if l.cnntranspose != nil {
		l.cnntranspose.SetBwdFiltAlgoPerformance(handle, bwdf)
		return
	}
}
func (l *layer) getcudnnperformance(handle *cudnn.Handler, x, y *layers.IO) (fwd []convolution.ForwardPerformance, bwddata []convolution.BackDataPerformance, bwdfilt []convolution.BackFilterPerformance, err error) {

	if l.cnn != nil {
		if debuglayerperformance {

		}
		fwd, err = l.cnn.GetFwdAlgoPerfList(handle, x, y)
		if err != nil {
			return nil, nil, nil, err
		}

		bwddata, err = l.cnn.GetBwdDataAlgoPerfList(handle, x, y)
		if err != nil {
			return nil, nil, nil, err
		}
		bwdfilt, err = l.cnn.GetBwdFiltAlgoPerfList(handle, x, y)
		if err != nil {
			return nil, nil, nil, err
		}
		return fwd, bwddata, bwdfilt, nil
	}
	if l.cnntranspose != nil {
		bwddata, err = l.cnntranspose.GetFwdAlgoPerfList(handle, x, y)
		if err != nil {
			return nil, nil, nil, err
		}

		fwd, err = l.cnntranspose.GetBwdDataAlgoPerfList(handle, x, y)
		if err != nil {
			return nil, nil, nil, err
		}
		bwdfilt, err = l.cnntranspose.GetBwdFiltAlgoPerfList(handle, x, y)
		if err != nil {
			return nil, nil, nil, err
		}
		return fwd, bwddata, bwdfilt, nil
	}
	return nil, nil, nil, nil
}
