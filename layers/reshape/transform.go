package reshape

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
)

func (l *Layer) transformtensforward(handle *cudnn.Handler, x, y *layers.IO) error {
	err := l.op.TransformForward(handle, l.defaultalpha, l.defaultbeta, x.T(), y.T())
	if err != nil {
		return err
	}
	return l.op.TransformForward(handle, l.defaultalpha, l.defaultbeta, x.DeltaT(), y.DeltaT())
}
func (l *Layer) transformtensbackward(handle *cudnn.Handler, x, y *layers.IO) error {
	err := l.op.TransformBackward(handle, l.defaultalpha, l.defaultbeta, x.T(), y.T())
	if err != nil {
		return err
	}
	return l.op.TransformBackward(handle, l.defaultalpha, l.defaultbeta, x.DeltaT(), y.DeltaT())
}
