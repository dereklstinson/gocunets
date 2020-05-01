package reshapes

import (
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/tensor"
)

//ResizeForward does a forward propagation resize channel dims and batch dim must equal
func (o *Ops) ResizeForward(handle *cudnn.Handler, x, y *tensor.Volume) error {
	return o.resize.ResizeForward(handle.XHandle(), x.TD(), x, y.TD(), y)
}

//ResizeBackward does a gradient propigation backwards resize channel dims and batch dim must equal
func (o *Ops) ResizeBackward(handle *cudnn.Handler, dx, dy *tensor.Volume) error {
	return o.resize.ResizeBackward(handle.XHandle(), dx.TD(), dx, dy.TD(), dy)
}
