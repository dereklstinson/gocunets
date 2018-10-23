package reshapes

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//ResizeForward does a forward propagation resize channel dims and batch dim must equal
func (o *Ops) ResizeForward(handle *gocudnn.XHandle, x, y *tensor.Volume) error {
	return o.resize.ResizeForward(handle, x.TD(), x.Memer(), y.TD(), y.Memer())
}

//ResizeBackward does a gradient propigation backwards resize channel dims and batch dim must equal
func (o *Ops) ResizeBackward(handle *gocudnn.XHandle, dx, dy *tensor.Volume) error {
	return o.resize.ResizeBackward(handle, dx.TD(), dx.Memer(), dy.TD(), dy.Memer())
}
