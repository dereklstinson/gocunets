package gocunets

/*
const debuglayerperformance = false

func (l *Layer) setcudnnperformancefwd(fwd convolution.ForwardPerformance) {
	if l.cnn != nil {
		l.cnn.SetFwdAlgoPerformance(fwd)
		return
	}
	if l.cnntranspose != nil {
		//Since cnntranspose runs backwards the forward prop is actually the backprop data algo.
		l.cnntranspose.SetBwdDataAlgoPerformance(fwd)
		return
	}
}
func (l *Layer) setcudnnperformancebwdd(bwdd convolution.BackDataPerformance) {
	if l.cnn != nil {
		l.cnn.SetBwdDataAlgoPerformance(bwdd)
		return
	}
	if l.cnntranspose != nil {
		//Since cnntranspose runs backwards the backprop for the data is actually the forward propagation algo
		l.cnntranspose.SetFwdAlgoPerformance(bwdd)
		return
	}
}
func (l *Layer) setcudnnperformancebwdf(bwdf convolution.BackFilterPerformance) {
	if l.cnn != nil {
		l.cnn.SetBwdFiltAlgoPerformance(bwdf)
		return
	}
	if l.cnntranspose != nil {
		l.cnntranspose.SetBwdFiltAlgoPerformance(bwdf)
		return
	}
}
func (l *Layer) getcudnnperformance(handle *cudnn.Handler, x, y *layers.Tensor, workspace *nvidia.Malloced) (fwd []convolution.ForwardPerformance, bwddata []convolution.BackDataPerformance, bwdfilt []convolution.BackFilterPerformance, err error) {

	if l.cnn != nil {
		if debuglayerperformance {

		}
		fwd, err = l.cnn.GetFwdAlgoPerfList(handle, x, y, workspace)
		if err != nil {
			return nil, nil, nil, err
		}

		bwddata, err = l.cnn.GetBwdDataAlgoPerfList(handle, x, y, workspace)
		if err != nil {
			return nil, nil, nil, err
		}
		bwdfilt, err = l.cnn.GetBwdFiltAlgoPerfList(handle, x, y, workspace)
		if err != nil {
			return nil, nil, nil, err
		}
		return fwd, bwddata, bwdfilt, nil
	}
	if l.cnntranspose != nil {
		bwddata, err = l.cnntranspose.GetFwdAlgoPerfList(handle, x, y, workspace)
		if err != nil {
			return nil, nil, nil, err
		}

		fwd, err = l.cnntranspose.GetBwdDataAlgoPerfList(handle, x, y, workspace)
		if err != nil {
			return nil, nil, nil, err
		}
		bwdfilt, err = l.cnntranspose.GetBwdFiltAlgoPerfList(handle, x, y, workspace)
		if err != nil {
			return nil, nil, nil, err
		}
		return fwd, bwddata, bwdfilt, nil
	}
	return nil, nil, nil, nil
}
*/
