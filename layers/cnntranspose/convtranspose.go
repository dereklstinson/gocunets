package cnntranspose

import (
	"github.com/dereklstinson/GoCuNets/cudnn/reshapes"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

/*
There is a few ways to do this.
1) using transformtensor and increase the size of the tensor. If done correctly you can make it so that every other value has 0.  //This will have to hold the output in mem
2) Using batch to shape then shape to batch.
3) Use resize and then do a resize back prop which back propigates the errors to the source pixel.
*/

type Layer struct {
	conv         *cnn.Layer
	trans        *reshapes.Ops
	mode         convtransposemode
	originaldims []int32
	outputdims   []int32
}
type convtransposemode int

const (
	convtransposetrans = convtransposemode(1)
)

//SetupTransform sets up a transform version of
func SetupTransform(handle *gocudnn.Handle,
	frmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
	space []int32, //space is the spacing between the elements of the input dims per dim if you want no space then put 0. Use the same frmt dims.
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	managedmem bool) (*Layer, error) {
	conv, err := cnn.SetupDynamic(handle, frmt, dtype, filterdims, filterdims, convmode, pad, stride, dilation, managedmem)
	if err != nil {
		return nil, err
	}

	return &Layer{
		conv: conv,
		mode: convtransposetrans,
	}, nil
}
