package reshapes

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//GetB2SOutputProperties returns the properties of the output
func (o *Ops) GetB2SOutputProperties(handle *cudnn.Handler, x *tensor.Volume, window, stride []int32) (frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32, err error) {
	if len(window) != 2 {
		return 255, 255, nil, errors.New("window can only have 2 elements")
	}

	frmt, dtype, dims, err = o.s2b.GetBatchtoShapeOutputProperties(x.TD(), window[0], window[1], stride[0], stride[1])

	return frmt, dtype, dims, err
}

//GetS2BOutputProperties returns the properties of the output
func (o *Ops) GetS2BOutputProperties(handle *cudnn.Handler, x *tensor.Volume, window, stride []int32) (frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32, err error) {
	if len(window) != 2 {
		return 255, 255, nil, errors.New("window can only have 2 elements")
	}

	frmt, dtype, dims, err = o.s2b.GetShapetoBatchOutputProperties(x.TD(), window[0], window[1], stride[0], stride[1])

	return frmt, dtype, dims, err

}

//GetS2BOutputPropertiesPLUS returns the properties of the output Pluss the n1 n2 used to increase the size of the batch
func (o *Ops) GetS2BOutputPropertiesPLUS(handle *cudnn.Handler, x *tensor.Volume, window, stride []int32) (frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32, n1n2 []int32, err error) {
	if len(window) != 2 {
		return 255, 255, nil, nil, errors.New("window can only have 2 elements")
	}

	frmt, dtype, dims, n1n2, err = o.s2b.GetShapetoBatchOutputPropertiesPLUS(x.TD(), window[0], window[1], stride[0], stride[1])

	return frmt, dtype, dims, n1n2, err

}

//S2BForward does changes the height and width of the 4d tensor x and places it into the batch dim of y.
func (o *Ops) S2BForward(handle *cudnn.Handler, x, y *tensor.Volume, stride []int32) error {
	return o.s2b.ShapeToBatch4d(handle.XHandle(), x.TD(), x.Memer(), y.TD(), y.Memer(), stride[0], stride[1], true)
}

//S2BBackward does the backward operation of space to batch aka batch to space values for y will go into x.
func (o *Ops) S2BBackward(handle *cudnn.Handler, x, y *tensor.Volume, stride []int32) error {
	return o.s2b.ShapeToBatch4d(handle.XHandle(), x.TD(), x.Memer(), y.TD(), y.Memer(), stride[0], stride[1], false)
}

//B2SForward will take multiple batches from x and place it into y
func (o *Ops) B2SForward(handle *cudnn.Handler, x, y *tensor.Volume, stride []int32) error {
	return o.s2b.ShapeToBatch4d(handle.XHandle(), y.TD(), y.Memer(), x.TD(), x.Memer(), stride[0], stride[1], false)
}

//B2SBackward will take the shapes from y and place it into the batches of x
func (o *Ops) B2SBackward(handle *cudnn.Handler, x, y *tensor.Volume, stride []int32) error {
	return o.s2b.ShapeToBatch4d(handle.XHandle(), y.TD(), y.Memer(), x.TD(), x.Memer(), stride[0], stride[1], true)
}
