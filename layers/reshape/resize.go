package reshape

import (
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//SpaceToBatchForwardProp does the forwardpropagation
func (l *Layer) resizeforward(handle *gocudnn.XHandle, x, y *layers.IO) error {
	err := l.op.ResizeForward(handle, x.T(), y.T())
	if err != nil {
		return err
	}
	return l.op.ResizeForward(handle, x.DeltaT(), y.DeltaT())
}

//SpaceToBatchBackward does the backward propagation
func (l *Layer) resizebackward(handle *gocudnn.XHandle, x, y *layers.IO) error {
	err := l.op.ResizeBackward(handle, x.T(), y.T())
	if err != nil {
		return err
	}
	return l.op.ResizeBackward(handle, x.DeltaT(), y.DeltaT())
}
