package cnntranspose

/*
This is old
import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers"
)

//GetBwdDataAlgoPerfList gets a list of forward performance algos
func (l *Layer) GetBwdDataAlgoPerfList(handle *cudnn.Handler, x, y *layers.Tensor, workspace *nvidia.Malloced) ([]convolution.ForwardPerformance, error) {
	return l.conv.GetReverseBwdDataAlgoPerfList(handle, x, y, workspace)
}

//GetFwdAlgoPerfList gets a list of backward performance algos
func (l *Layer) GetFwdAlgoPerfList(handle *cudnn.Handler, x, y *layers.Tensor, workspace *nvidia.Malloced) ([]convolution.BackDataPerformance, error) {
	return l.conv.GetReverseFwdAlgoPerfList(handle, x, y, workspace)
}

//GetBwdFiltAlgoPerfList gets a list of forward performance algos
func (l *Layer) GetBwdFiltAlgoPerfList(handle *cudnn.Handler, x, y *layers.Tensor, workspace *nvidia.Malloced) ([]convolution.BackFilterPerformance, error) {
	return l.conv.GetReverseBwdFiltAlgoPerfList(handle, x, y, workspace)
}

//SetBwdDataAlgoPerformance sets the Performance Values
func (l *Layer) SetBwdDataAlgoPerformance(bwddata convolution.ForwardPerformance) {
	l.conv.SetReverseBwdDataAlgoPerformance(bwddata)
}

//SetBwdFiltAlgoPerformance sets the Performance Values
func (l *Layer) SetBwdFiltAlgoPerformance(bwdfilt convolution.BackFilterPerformance) {
	l.conv.SetReverseBwdFiltAlgoPerformance(bwdfilt)
}

//SetFwdAlgoPerformance sets the Performance Values
func (l *Layer) SetFwdAlgoPerformance(fwd convolution.BackDataPerformance) {
	l.conv.SetReverseFwdAlgoPerformance(fwd)
}
*/
