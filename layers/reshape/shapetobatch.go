package reshape

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
)

//GetShapetoBatchIO will return the output IO for the S2B op.
func (l *Layer) getshapetobatchio(handle *cudnn.Handler, x *layers.IO, input bool) (*layers.IO, error) {
	yfrmt, ydtype, dims, err := l.op.GetS2BOutputProperties(handle, x.T(), l.window, l.stride)
	if err != nil {

		return nil, err
	}
	if input == false {
		return layers.BuildIO(handle, cudnn.TensorFormat(yfrmt), cudnn.DataType(ydtype), dims)
	}
	return layers.BuildNetworkInputIO(handle, cudnn.TensorFormat(yfrmt), cudnn.DataType(ydtype), dims)
}

//GetShapetoBatchIO will return the output IO for the S2B op.
func (l *Layer) getshapetobatchioinference(handle *cudnn.Handler, x *layers.IO, input bool) (*layers.IO, error) {
	yfrmt, ydtype, dims, err := l.op.GetS2BOutputProperties(handle, x.T(), l.window, l.stride)
	if err != nil {
		return nil, err
	}
	return layers.BuildInferenceIO(handle, cudnn.TensorFormat(yfrmt), cudnn.DataType(ydtype), dims)
}

//SpaceToBatchForwardProp does the forwardpropagation
func (l *Layer) spacetobatchforwardprop(handle *cudnn.Handler, x, y *layers.IO) error {
	err := l.op.S2BForward(handle, x.T(), y.T(), l.stride)
	if err != nil {
		return err
	}
	return l.op.S2BForward(handle, x.DeltaT(), y.DeltaT(), l.stride)
}

//SpaceToBatchBackward does the backward propagation
func (l *Layer) spacetobatchbackprop(handle *cudnn.Handler, x, y *layers.IO) error {
	err := l.op.S2BBackward(handle, x.T(), y.T(), l.stride)
	if err != nil {
		return err
	}
	return l.op.S2BBackward(handle, x.DeltaT(), y.DeltaT(), l.stride)
}
