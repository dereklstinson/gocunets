package dropout

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/dropout"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer holds the op for the dropout
type Layer struct {
	op *dropout.Ops
}
type Settings struct {
	Dropout float32 `json:"dropout,omitempty"`
	Seed    uint64  `json:"seed,omitempty"`
	Managed bool    `json:"managed,omitempty"`
}

//LayerSetup sets up the layer
func Setup(handle *gocudnn.Handle, x *layers.IO, drpout float32, seed uint64, managed bool) (*Layer, error) {
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
