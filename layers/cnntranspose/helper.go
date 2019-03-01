package cnntranspose

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers"
)

//GetBwdDataAlgoPerfList gets a list of forward performance algos
func (l *Layer) GetBwdDataAlgoPerfList(handle *cudnn.Handler, x, y *layers.IO, workspace *nvidia.Malloced) ([]convolution.ForwardPerformance, error) {
	return l.conv.GetReverseBwdDataAlgoPerfList(handle, x, y, workspace)
}

//GetFwdAlgoPerfList gets a list of backward performance algos
func (l *Layer) GetFwdAlgoPerfList(handle *cudnn.Handler, x, y *layers.IO, workspace *nvidia.Malloced) ([]convolution.BackDataPerformance, error) {
	return l.conv.GetReverseFwdAlgoPerfList(handle, x, y, workspace)
}

//GetBwdFiltAlgoPerfList gets a list of forward performance algos
func (l *Layer) GetBwdFiltAlgoPerfList(handle *cudnn.Handler, x, y *layers.IO, workspace *nvidia.Malloced) ([]convolution.BackFilterPerformance, error) {
	return l.conv.GetReverseBwdFiltAlgoPerfList(handle, x, y, workspace)
}

//SetBwdDataAlgoPerformance sets the Performance Values
func (l *Layer) SetBwdDataAlgoPerformance(handle *cudnn.Handler,
	bwddata convolution.ForwardPerformance) {
	l.conv.SetReverseBwdDataAlgoPerformance(handle, bwddata)
}

//SetBwdFiltAlgoPerformance sets the Performance Values
func (l *Layer) SetBwdFiltAlgoPerformance(handle *cudnn.Handler,
	bwdfilt convolution.BackFilterPerformance) {
	l.conv.SetReverseBwdFiltAlgoPerformance(handle, bwdfilt)
}

//SetFwdAlgoPerformance sets the Performance Values
func (l *Layer) SetFwdAlgoPerformance(handle *cudnn.Handler,
	fwd convolution.BackDataPerformance) {
	l.conv.SetReverseFwdAlgoPerformance(handle, fwd)
}