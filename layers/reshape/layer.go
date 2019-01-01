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

//SetupB2S sets up Batch 2 Shape layer.
//If networkinput is true the delta values will not be passed.
//Window decides how it will shape with respect to h and w. Window must be a factor of the batch.
//Example NCHW vector of [24,5,2,3].  If window of [4,3]. The vector output will be [2,5,8,9].
//Placing batches is row dominant like in C.
func SetupB2S(handle *cudnn.Handler, window []int32, networkinput bool) (*Layer, error) {
	op, err := reshapes.Stage(handle)
	var lmf ModeFlag
	mode := lmf.B2S()
	return &Layer{op: op, mode: mode, window: window, networkinput: networkinput, defaultalpha: defaultalpha, defaultbeta: defaultbeta}, err
}

//SetupS2B sets up Shape 2 Batch layer.
//If networkinput is true the delta values will not be passed.
//Window decides how it will shape of the batches with h and w.  The window doesn't need to be a factor of the input values. The last values will be zero
//Example NCHW vector of [2,5,8,8].  If window of [3,3]. The vector output will be [18,5,3,3].
//Placing batches is row dominant like in C.
func SetupS2B(handle *cudnn.Handler, window []int32, networkinput bool) (*Layer, error) {
	op, err := reshapes.Stage(handle)
	var lmf ModeFlag
	mode := lmf.S2B()
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
	case lmf.B2S():
		return l.getbatchtoshapeio(handle, x, l.window[0], l.window[1], l.networkinput)
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
	case lmf.B2S():
		return l.batchtoshapeforwardprop(handle, x, y)
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
	case lmf.B2S():
		return l.batchtoshapebackprop(handle, x, y)
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

//B2S is batch to shape
func (l ModeFlag) B2S() Mode {
	return Mode(5)
}
