package reshape

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
)

//GetShapetoBatchIO will return the output IO for the S2B op.
func (l *Layer) getbatchtoshapeio(handle *cudnn.Handler, x *layers.Tensor, input bool) (*layers.Tensor, error) {

	yfrmt, ydtype, dims, err := l.op.GetB2SOutputProperties(x.Volume, l.stride, l.window)

	if err != nil {

		return nil, err
	}

	return layers.CreateTensor(handle, (yfrmt), (ydtype), dims)
}

//GetShapetoBatchIO will return the output IO for the S2B op.
func (l *Layer) getbatchtoshapeioinference(handle *cudnn.Handler, x *layers.Tensor, input bool) (*layers.Tensor, error) {

	yfrmt, ydtype, dims, err := l.op.GetB2SOutputProperties(x.Volume, l.stride, l.window)

	if err != nil {

		return nil, err
	}

	return layers.CreateTensor(handle, (yfrmt), (ydtype), dims)
}

//SpaceToBatchForwardProp does the forwardpropagation
func (l *Layer) batchtoshapeforwardprop(handle *cudnn.Handler, x, y *layers.Tensor) error {
	err := l.op.B2SForward(handle, x.Volume, y.Volume, l.stride)
	if err != nil {
		return err
	}
	return nil
}

//SpaceToBatchBackward does the backward propagation
func (l *Layer) batchtoshapebackprop(handle *cudnn.Handler, x, y *layers.Tensor) error {
	err := l.op.B2SBackward(handle, x.Volume, y.Volume, l.stride)
	if err != nil {
		return err
	}
	return nil
}
