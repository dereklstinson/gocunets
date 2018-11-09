package dropout

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/dropout"
	"github.com/dereklstinson/GoCuNets/layers"
)

//Layer holds the op for the dropout
type Layer struct {
	op *dropout.Ops
}

//Settings is the settings for drop out layer
type Settings struct {
	Dropout float32 `json:"dropout,omitempty"`
	Seed    uint64  `json:"seed,omitempty"`
	Managed bool    `json:"managed,omitempty"`
}

//Setup sets up the layer
func Setup(handle *cudnn.Handler, x *layers.IO, drpout float32, seed uint64, managed bool) (*Layer, error) {
	op, err := dropout.Stage(handle, x.T(), drpout, seed, managed)
	return &Layer{
		op: op,
	}, err
}

//ForwardProp does the forward propagation
func (l *Layer) ForwardProp(handle *cudnn.Handler, x, y *layers.IO) error {
	return l.op.ForwardProp(handle, x.T(), y.T())
}

//BackProp does the back propagation
func (l *Layer) BackProp(handle *cudnn.Handler, x, y *layers.IO) error {
	return l.op.BackProp(handle, x.DeltaT(), y.DeltaT())
}
