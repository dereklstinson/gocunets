package reshapes

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//GetB2SOutputProperties returns the properties of the output
func (o *Ops) GetB2SOutputProperties(handle *cudnn.Handler, x *tensor.Volume, window []int32) (gocudnn.TensorFormat, gocudnn.DataType, []int32, bool, error) {
	if len(window) != 2 {
		return 255, 255, nil, false, errors.New("window can only have 2 elements")
	}

	xmal := x.Memer()
	if xmal != nil {
		var managed bool
		var flgloc gocudnn.LocationFlag
		if flgloc.Unified() == xmal.Stored() {
			managed = true
		}
		frmt, dtype, dims, err := o.s2b.GetBatchtoShapeOutputProperties(x.TD(), window[0], window[1])

		return frmt, dtype, dims, managed, err
	}

	return 255, 255, nil, false, errors.New("memory is nil")
}

//GetS2BOutputProperties returns the properties of the output
func (o *Ops) GetS2BOutputProperties(handle *cudnn.Handler, x *tensor.Volume, window []int32) (gocudnn.TensorFormat, gocudnn.DataType, []int32, bool, error) {
	if len(window) != 2 {
		return 255, 255, nil, false, errors.New("window can only have 2 elements")
	}

	xmal := x.Memer()
	if xmal != nil {
		var managed bool
		var flgloc gocudnn.LocationFlag
		if flgloc.Unified() == xmal.Stored() {
			managed = true
		}
		frmt, dtype, dims, err := o.s2b.GetShapetoBatchOutputProperties(x.TD(), window[0], window[1])

		return frmt, dtype, dims, managed, err

	}

	return 255, 255, nil, false, errors.New("memory is nil")
}

//S2BForward does changes the height and width of the 4d tensor x and places it into the batch dim of y.
func (o *Ops) S2BForward(handle *cudnn.Handler, x, y *tensor.Volume) error {
	return o.s2b.ShapeToBatch4d(handle.XHandle(), x.TD(), x.Memer(), y.TD(), y.Memer(), true)
}

//S2BBackward does the backward operation of space to batch aka batch to space values for y will go into x.
func (o *Ops) S2BBackward(handle *cudnn.Handler, x, y *tensor.Volume) error {
	return o.s2b.ShapeToBatch4d(handle.XHandle(), x.TD(), x.Memer(), y.TD(), y.Memer(), false)
}
