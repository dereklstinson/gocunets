package reshape

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
)

//GetShapetoBatchIO will return the output IO for the S2B op.
func (l *Layer) getbatchtoshapeio(handle *cudnn.Handler, x *layers.IO, input bool) (*layers.IO, error) {

	yfrmt, ydtype, dims, err := l.op.GetB2SOutputProperties(handle, x.T(), l.stride, l.window)

	if err != nil {

		return nil, err
	}
	if input == false {
		return layers.BuildIO(handle, cudnn.TensorFormat(yfrmt), cudnn.DataType(ydtype), dims)
	}
	return layers.BuildNetworkInputIO(handle, cudnn.TensorFormat(yfrmt), cudnn.DataType(ydtype), dims)
}

//SpaceToBatchForwardProp does the forwardpropagation
func (l *Layer) batchtoshapeforwardprop(handle *cudnn.Handler, x, y *layers.IO) error {
	err := l.op.B2SForward(handle, x.T(), y.T(), l.stride)
	if err != nil {
		return err
	}
	return l.op.B2SForward(handle, x.DeltaT(), y.DeltaT(), l.stride)
}

//SpaceToBatchBackward does the backward propagation
func (l *Layer) batchtoshapebackprop(handle *cudnn.Handler, x, y *layers.IO) error {
	err := l.op.B2SBackward(handle, x.T(), y.T(), l.stride)
	if err != nil {
		return err
	}
	return l.op.B2SBackward(handle, x.DeltaT(), y.DeltaT(), l.stride)
}
