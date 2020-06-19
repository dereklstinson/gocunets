package dropout

import (
	"errors"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/dropout"
	"github.com/dereklstinson/gocunets/layers"
)

//Layer holds the op for the dropout
type Layer struct {
	op      *dropout.Ops
	dropout float32
	seed    uint64
}

//Settings is the settings for drop out layer
type Settings struct {
	Dropout float32 `json:"dropout,omitempty"`
	Seed    uint64  `json:"seed,omitempty"`
}

//Setup sets up the layer
func Setup(handle *cudnn.Handler, x *layers.Tensor, drpout float32, seed uint64) (*Layer, error) {
	op, err := dropout.Stage(handle, x.Volume, drpout, seed)
	return &Layer{
		op: op,
	}, err
}

//Preset presets the layer, but doesn't build it. Useful if you want to set up the network before the tensor descriptors for the input
//are made. handle can be nil.  I just wanted to keep it consistant.
func Preset(handle *cudnn.Handler, dropout float32, seed uint64) (*Layer, error) {
	if dropout >= 1 {
		return nil, errors.New("Dropout can't be greater than or equal to 1")
	}
	return &Layer{
		dropout: dropout,
		seed:    seed,
	}, nil
}

//BuildFromPreset will construct the dropout layer if preset was made. Both handle and input are needed.
//This can also be called again if input size has changed
func (l *Layer) BuildFromPreset(handle *cudnn.Handler, input *layers.Tensor) error {
	var err error

	l.op, err = dropout.Stage(handle, input.Volume, l.dropout, l.seed)
	return err
}

//ForwardProp does the forward propagation
func (l *Layer) ForwardProp(handle *cudnn.Handler, x, y *layers.Tensor) error {
	return l.op.ForwardProp(handle, x.Volume, y.Volume)
}

//BackProp does the back propagation
func (l *Layer) BackProp(handle *cudnn.Handler, dx, dy *layers.Tensor) error {
	return l.op.BackProp(handle, dx.Volume, dy.Volume)
}
