package reshapes

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	"github.com/dereklstinson/GoCudnn"
)

//TransformForward fills tensor y with x to the best of its ability.
func (o *Ops) TransformForward(handle *cudnn.Handler, alpha, beta float64, x, y *tensor.Volume) error {
	dtype := x.TD().DataType()
	a := gocudnn.CScalarByDataType(dtype, alpha)
	b := gocudnn.CScalarByDataType(dtype, beta)

	return gocudnn.Tensor{}.TransformTensor(handle.Cudnn(), a, x.TDStrided(), x.Memer(), b, y.TDStrided(), y.Memer())
}

//TransformBackward fills tensor x with the values of y to the best of its ability
func (o *Ops) TransformBackward(handle *cudnn.Handler, alpha, beta float64, x, y *tensor.Volume) error {
	return o.TransformForward(handle, alpha, beta, y, x)
}
