package dropout

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor/dropout"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer holds the op for the dropout
type Layer struct {
	op *dropout.Ops
}

//LayerSetup sets up the layer
func LayerSetup(handle *gocudnn.Handle, x *layers.IO, drpout float32, seed uint64, managed bool) (*Layer, error) {
	op, err := dropout.Stage(handle, x.T(), drpout, seed, managed)
	return &Layer{
		op: op,
	}, err
}
func (l *Layer) ForwardProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	return l.op.ForwardProp(handle, x.T(), y.T())
}
func (l *Layer) BackProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	return l.op.BackProp(handle, x.DeltaT(), y.DeltaT())
}
