package reshape

import (
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCudnn"
)

//GetShapetoBatchIO will return the output IO for the S2B op.
func (l *Layer) GetShapetoBatchIO(handle *gocudnn.XHandle, x *layers.IO, h, w int32) (*layers.IO, error) {
	y, err := l.op.GetS2BOutputVolume(handle, x.T(), []int32{h, w})
	if err != nil {

		return nil, err
	}
	dy, err := l.op.GetS2BOutputVolume(handle, x.DeltaT(), []int32{h, w})
	if err != nil {
		y.Destroy()
		return nil, err
	}
	return layers.CreateIOfromVolumes(y, dy)
}

//SpaceToBatchForwardProp does the forwardpropagation
func (l *Layer) SpaceToBatchForwardProp(handle *gocudnn.XHandle, x, y *layers.IO) error {
	err := l.op.S2BForward(handle, x.T(), y.T())
	if err != nil {
		return err
	}
	return l.op.S2BForward(handle, x.DeltaT(), y.DeltaT())
}

//SpaceToBatchBackward does the backward propagation
func (l *Layer) SpaceToBatchBackward(handle *gocudnn.XHandle, x, y *layers.IO) error {
	err := l.op.S2BBackward(handle, x.T(), y.T())
	if err != nil {
		return err
	}
	return l.op.S2BBackward(handle, x.DeltaT(), y.DeltaT())
}
