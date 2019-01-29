package reshape

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/Nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
)

//GetTransposeIO will return an IO that can be used with the transpose function of this layer.
func (l *Layer) gettransposeIO(handle *cudnn.Handler, x *layers.IO, input bool) (*layers.IO, error) {
	yfrmt, ydtype, dims, _, err := l.op.GetTransposeOutputProperties(handle, x.T())
	if err != nil {
		return nil, err
	}

	if input == false {
		return layers.BuildIO(handle, yfrmt, ydtype, dims)
	}
	return layers.BuildNetworkInputIO(handle, yfrmt, ydtype, dims)
}

//TransposeForward does a transpose allong the channel dim. Will find transpose of x and put it in y
func (l *Layer) transposeforwardprop(handle *cudnn.Handler, x, y *layers.IO) error {
	err := l.op.TransposeChannelForward(handle, x.T(), y.T())
	if err != nil {
		return err
	}
	return l.op.TransposeChannelForward(handle, x.DeltaT(), y.DeltaT())
}

//TransposeBackward does a transpose allong the channel dim. Will find transpose of y and put it into x
func (l *Layer) transposebackprop(handle *cudnn.Handler, x, y *layers.IO) error {
	err := l.op.TransposeChannelBackward(handle, x.T(), y.T())
	if err != nil {
		return err
	}
	return l.op.TransposeChannelBackward(handle, x.DeltaT(), y.DeltaT())
}
