package reshape

import (
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/custom/reshapes"
	"github.com/dereklstinson/gocunets/layers"
)

func (l *Layer) transformtensforward(handle *cudnn.Handler, x, y *layers.Tensor, hlpr *TransFormHelper) error {
	err := l.op.TransformForward(handle, l.defaultalpha, l.defaultbeta, x.Volume, y.Volume, hlpr.hlpr)
	if err != nil {
		return err
	}
	return nil
}
func (l *Layer) transformtensbackward(handle *cudnn.Handler, x, y *layers.Tensor, hlpr *TransFormHelper) error {
	err := l.op.TransformBackward(handle, l.defaultalpha, l.defaultbeta, x.Volume, y.Volume, hlpr.hlpr)
	if err != nil {
		return err
	}
	return nil
}

//TransFormHelper helps reshaping
type TransFormHelper struct {
	hlpr *reshapes.TransFormHelper
}

//MakeTranFormHelper create a transformhelper
func (l *Layer) MakeTranFormHelper(x, y *layers.Tensor) (*TransFormHelper, error) {
	helper, err := l.op.MakeTransformHelper(x.Volume, y.Volume)
	if err != nil {
		return nil, err
	}
	return &TransFormHelper{
		hlpr: helper,
	}, nil
}
