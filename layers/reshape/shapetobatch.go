package reshape

import (
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCudnn"
)

//GetShapetoBatchIO will return the output IO for the S2B op.
func (l *Layer) getshapetobatchio(handle *gocudnn.XHandle, x *layers.IO, h, w int32, input bool) (*layers.IO, error) {
	yfrmt, ydtype, dims, managed, err := l.op.GetS2BOutputProperties(handle, x.T(), []int32{h, w})
	if err != nil {

		return nil, err
	}
	if input == false {
		return layers.BuildIO(yfrmt, ydtype, dims, managed)
	}
	return layers.BuildNetworkInputIO(yfrmt, ydtype, dims, managed)
}

//SpaceToBatchForwardProp does the forwardpropagation
func (l *Layer) spacetobatchforwardprop(handle *gocudnn.XHandle, x, y *layers.IO) error {
	err := l.op.S2BForward(handle, x.T(), y.T())
	if err != nil {
		return err
	}
	return l.op.S2BForward(handle, x.DeltaT(), y.DeltaT())
}

//SpaceToBatchBackward does the backward propagation
func (l *Layer) spacetobatchbackprop(handle *gocudnn.XHandle, x, y *layers.IO) error {
	err := l.op.S2BBackward(handle, x.T(), y.T())
	if err != nil {
		return err
	}
	return l.op.S2BBackward(handle, x.DeltaT(), y.DeltaT())
}
