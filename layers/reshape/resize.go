package reshape

import (
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/layers"
)

//SpaceToBatchForwardProp does the forwardpropagation
func (l *Layer) resizeforward(handle *cudnn.Handler, x, y *layers.Tensor) error {
	err := l.op.ResizeForward(handle, x.Volume, y.Volume)
	if err != nil {
		return err
	}
	return nil
}

//SpaceToBatchBackward does the backward propagation
func (l *Layer) resizebackward(handle *cudnn.Handler, x, y *layers.Tensor) error {
	err := l.op.ResizeBackward(handle, x.Volume, y.Volume)
	if err != nil {
		return err
	}
	return nil
}
