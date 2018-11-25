package reshape

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/reshapes"
	"github.com/dereklstinson/GoCuNets/layers"
)

//Layer is the that type that handles reshape methods
type Layer struct {
	op                        *reshapes.Ops
	mode                      Mode
	window                    []int32
	networkinput              bool
	defaultalpha, defaultbeta float64
}

const defaultalpha = float64(1.0)
const defaultbeta = float64(1.0)

//Build builds the layer mode picks the mode window has to be passed if S2B it is a set size that you want each batch to be.
//window will be ignored if mode was picked to transpose
//All will be ignored of you pick transform
func Build(handle *cudnn.Handler, mode Mode, window []int32, networkinput bool) (*Layer, error) {
	//var lmf ModeFlag

	op, err := reshapes.Stage(handle)
	return &Layer{op: op, mode: mode, window: window, networkinput: networkinput, defaultalpha: defaultalpha, defaultbeta: defaultbeta}, err
}

//MakeOutputTensor returns a layer.IO for the network
func (l *Layer) MakeOutputTensor(handle *cudnn.Handler, x *layers.IO) (*layers.IO, error) {
	var lmf ModeFlag
	switch l.mode {
	case lmf.Transpose():
		return l.gettransposeIO(handle, x, l.networkinput)
	case lmf.S2B():
		return l.getshapetobatchio(handle, x, l.window[0], l.window[1], l.networkinput)
	case lmf.Resize():
		return nil, errors.New("No output descriptor needed for resize")
	case lmf.Transform():
		return nil, errors.New("No output descriptor needed for transform")
	}

	return nil, errors.New("Layer doesn't support mode passed")
}

//ForwardProp performs the forward prop x is the input and y is the input and output
func (l *Layer) ForwardProp(handle *cudnn.Handler, x, y *layers.IO) error {
	var lmf ModeFlag
	switch l.mode {
	case lmf.Transpose():
		return l.transposeforwardprop(handle, x, y)
	case lmf.S2B():
		return l.spacetobatchforwardprop(handle, x, y)
	case lmf.Resize():
		return l.resizeforward(handle, x, y)
	case lmf.Transform():
		//return l.transformtensforward(handle, x, y)
		return errors.New("THIS NEEDS FIXED")
	}

	return errors.New("Layer doesn't support mode passed")
}

//BackProp performs the backprop prop x is the input and output and y is the input
func (l *Layer) BackProp(handle *cudnn.Handler, x, y *layers.IO) error {
	var lmf ModeFlag
	switch l.mode {
	case lmf.Transpose():
		return l.transposebackprop(handle, x, y)
	case lmf.S2B():
		return l.spacetobatchbackprop(handle, x, y)
	case lmf.Resize():
		return l.resizebackward(handle, x, y)
	case lmf.Transform():
		//	return l.transformtensbackward(handle, x, y)
		return errors.New("THIS NEEDS FIXED")
	}

	return errors.New("Layer doesn't support mode passed")
}

//Mode is a flag set for this layer
type Mode int

//ModeFlag passes flags for this layer through methods.
type ModeFlag struct {
}

//Transpose sets layer mode to transpose
func (l ModeFlag) Transpose() Mode {
	return Mode(1)
}

//S2B sets layer mode to space to batch
func (l ModeFlag) S2B() Mode {
	return Mode(2)
}

//Resize sets layer mode to space to batch
func (l ModeFlag) Resize() Mode {
	return Mode(3)
}

//Transform performs the transform op
func (l ModeFlag) Transform() Mode {
	return Mode(4)
}
