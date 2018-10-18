package reshape

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/gocudnn/reshapes"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCudnn"
)

//Layer is the that type that handles reshape methods
type Layer struct {
	op           *reshapes.Ops
	mode         LayerMode
	window       []int32
	networkinput bool
}

//Build builds the layer mode picks the mode window has to be passed if S2B it is a set size that you want each batch to be.
//window will be ignored if mode was picked to transpose
func Build(handle *gocudnn.XHandle, mode LayerMode, window []int32, networkinput bool) (*Layer, error) {
	//var lmf LayerModeFlag

	op, err := reshapes.Stage(handle)
	return &Layer{op: op, mode: mode, window: window, networkinput: networkinput}, err
}

//MakeOutputTensor returns a layer.IO for the network
func (l *Layer) MakeOutputTensor(handle *gocudnn.XHandle, x *layers.IO) (*layers.IO, error) {
	var lmf LayerModeFlag
	switch l.mode {
	case lmf.Transpose():
		return l.gettransposeIO(handle, x, l.networkinput)
	case lmf.S2B():
		return l.getshapetobatchio(handle, x, l.window[0], l.window[1], l.networkinput)
	}
	return nil, errors.New("Layer doesn't support mode passed")
}

//ForwardProp performs the forward prop x is the input and y is the input and output
func (l *Layer) ForwardProp(handle *gocudnn.XHandle, x, y *layers.IO) error {
	var lmf LayerModeFlag
	switch l.mode {
	case lmf.Transpose():
		return l.transposeforwardprop(handle, x, y)
	case lmf.S2B():
		return l.spacetobatchforwardprop(handle, x, y)
	}
	return errors.New("Layer doesn't support mode passed")
}

//BackProp performs the backprop prop x is the input and output and y is the input
func (l *Layer) BackProp(handle *gocudnn.XHandle, x, y *layers.IO) error {
	var lmf LayerModeFlag
	switch l.mode {
	case lmf.Transpose():
		return l.transposebackprop(handle, x, y)
	case lmf.S2B():
		return l.spacetobatchbackprop(handle, x, y)
	}
	return errors.New("Layer doesn't support mode passed")
}

//LayerMode is a flag set for this layer
type LayerMode int

//LayerModeFlag passes flags for this layer through methods.
type LayerModeFlag struct {
}

//Transpose sets layer mode to transpose
func (l LayerModeFlag) Transpose() LayerMode {
	return LayerMode(1)
}

//S2B sets layer mode to space to batch
func (l LayerModeFlag) S2B() LayerMode {
	return LayerMode(2)
}
