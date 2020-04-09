package reshape

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
)

//GetShapetoBatchIO will return the output IO for the S2B op.
func (l *Layer) getshapetobatchio(handle *cudnn.Handler, x *layers.Tensor, input bool) (*layers.Tensor, error) {
	yfrmt, ydtype, dims, err := l.op.GetS2BOutputProperties(x.Volume, l.window, l.stride)
	if err != nil {

		return nil, err
	}

	return layers.CreateTensor(handle, (yfrmt), (ydtype), dims)
}

//GetShapetoBatchIO will return the output IO for the S2B op.
func (l *Layer) getshapetobatchioinference(handle *cudnn.Handler, x *layers.Tensor, input bool) (*layers.Tensor, error) {
	yfrmt, ydtype, dims, err := l.op.GetS2BOutputProperties(x.Volume, l.window, l.stride)
	if err != nil {
		return nil, err
	}
	return layers.CreateTensor(handle, (yfrmt), (ydtype), dims)
}

//SpaceToBatchForwardProp does the forwardpropagation
func (l *Layer) spacetobatchforwardprop(handle *cudnn.Handler, x, y *layers.Tensor) error {
	return l.op.S2BForward(handle, x.Volume, y.Volume, l.stride)

}

//SpaceToBatchBackward does the backward propagation
func (l *Layer) spacetobatchbackprop(handle *cudnn.Handler, x, y *layers.Tensor) error {
	return l.op.S2BBackward(handle, x.Volume, y.Volume, l.stride)

}
