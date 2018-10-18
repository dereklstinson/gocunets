package reshapes

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//GetS2BOutputVolume returns a volume for the output
func (o *Ops) GetS2BOutputProperties(handle *gocudnn.XHandle, x *tensor.Volume, window []int32) (gocudnn.TensorFormat, gocudnn.DataType, []int32, bool, error) {
	if len(window) != 2 {
		return 255, 255, nil, false, errors.New("window can only have 2 elements")
	}

	xmal, ok := x.Memer().(*gocudnn.Malloced)
	if ok {
		var managed bool
		var flgloc gocudnn.LocationFlag
		if flgloc.Unified() == xmal.Stored() {
			managed = true
		}
		frmt, dtype, dims, err := o.s2b.GetShapetoBatchOutputProperties(x.TD(), window[0], window[1])

		return frmt, dtype, dims, managed, err
	}

	return 255, 255, nil, false, errors.New("Unsupported Format of Memer for S2B")
}

//S2BForward does changes the height and width of the 4d tensor x and places it into the batch dim of y.
func (o *Ops) S2BForward(handle *gocudnn.XHandle, x, y *tensor.Volume) error {
	return o.s2b.ShapeToBatch4d(handle, x.TD(), x.Memer(), y.TD(), y.Memer(), true)
}

//S2BBackward does the backward operation of space to batch aka batch to space values for y will go into x.
func (o *Ops) S2BBackward(handle *gocudnn.XHandle, x, y *tensor.Volume) error {
	return o.s2b.ShapeToBatch4d(handle, x.TD(), x.Memer(), y.TD(), y.Memer(), false)
}
