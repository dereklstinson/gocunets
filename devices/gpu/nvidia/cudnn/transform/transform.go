package transform

import (
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/gocudnn"
)

//Ops is the struct that holds the descriptor for the transform operation
type Ops struct {
	op *gocudnn.TransformD
}

//Stage stages transform operation
func Stage(nbDims uint32, destfrmt gocudnn.TensorFormat, padbefore, padafter []int32, fold []uint32, direction gocudnn.FoldingDirection) (op *Ops, err error) {
	op.op, err = gocudnn.CreateTransformDescriptor()
	if err != nil {
		return nil, err
	}
	err = op.op.Set(nbDims, destfrmt, padbefore, padafter, fold, direction)
	return op, err
}

//Transform does the transform operation
func (t *Ops) Transform(handle cudnn.Handler, alpha float64, src *tensor.Volume, beta float64, dest *tensor.Volume) error {
	return t.op.TransformTensor(handle.Cudnn(), alpha, src.TD(), src.Memer(), beta, dest.TD(), dest.Memer())
}
