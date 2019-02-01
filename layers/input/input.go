package input

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/Nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/Nvidia/custom/tconv"
	"github.com/dereklstinson/GoCuNets/layers"
)

//Layer holds the type uint8tofloat conversion conversion operation
type Layer struct {
	t          *tconv.Op
	normalized bool
}

//Stage stages the input layer. normalize will divide all the values of the input by 255 after it has been converted to a float.
func Stage(handle *cudnn.Handler, normalize bool) (*Layer, error) {
	t, err := tconv.CreateInt8ToFloat(handle)
	if err != nil {
		return nil, err
	}
	return &Layer{
		t:          t,
		normalized: normalize,
	}, nil
}

//GetOutputIO returns the output IO
func (l *Layer) GetOutputIO(handle *cudnn.Handler, input *layers.IO) (*layers.IO, error) {
	var dflg cudnn.DataTypeFlag
	frmt, dtype, dims, err := input.Properties()
	if err != nil {
		return nil, err
	}
	if dtype != dflg.Int8() {
		return nil, errors.New("Input Datatype needs to be int8")
	}

	return layers.BuildNetworkInputIO(handle, frmt, dflg.Float(), dims)
}

//Convert converts the values of x into float32 in y
func (l *Layer) Convert(handle *cudnn.Handler, x, y *layers.IO) error {
	return l.t.Int8ToFloat(handle, x.T(), y.T(), l.normalized)
}
