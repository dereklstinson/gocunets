package reshape

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/custom/reshapes"
	"github.com/dereklstinson/GoCuNets/layers"
)

func (l *Layer) transformtensforward(handle *cudnn.Handler, x, y *layers.IO, hlpr *TransFormHelper) error {
	err := l.op.TransformForward(handle, l.defaultalpha, l.defaultbeta, x.T(), y.T(), hlpr.hlpr)
	if err != nil {
		return err
	}
	return l.op.TransformForward(handle, l.defaultalpha, l.defaultbeta, x.DeltaT(), y.DeltaT(), hlpr.hlpr)
}
func (l *Layer) transformtensbackward(handle *cudnn.Handler, x, y *layers.IO, hlpr *TransFormHelper) error {
	err := l.op.TransformBackward(handle, l.defaultalpha, l.defaultbeta, x.T(), y.T(), hlpr.hlpr)
	if err != nil {
		return err
	}
	return l.op.TransformBackward(handle, l.defaultalpha, l.defaultbeta, x.DeltaT(), y.DeltaT(), hlpr.hlpr)
}

type TransFormHelper struct {
	hlpr *reshapes.TransFormHelper
}

func (l *Layer) MakeTranFormHelper(x, y *layers.IO) (*TransFormHelper, error) {
	helper, err := l.op.MakeTransformHelper(x.T(), y.T())
	if err != nil {
		return nil, err
	}
	return &TransFormHelper{
		hlpr: helper,
	}, nil
}
