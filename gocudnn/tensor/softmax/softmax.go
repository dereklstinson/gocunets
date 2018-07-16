package softmax

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	"github.com/dereklstinson/GoCudnn"
)

//SoftMax does the softmax algo
type SoftMax struct {
	algo   gocudnn.SoftMaxAlgorithm
	helper gocudnn.SoftMax
	mode   gocudnn.SoftMaxMode
	alpha  gocudnn.CScalar
	beta   gocudnn.CScalar
}

//BuildDefault builds a default layer (only option for now)
func BuildDefault() *SoftMax {
	var s gocudnn.SoftMax
	//	fmt, dtype, dims, err := input.Properties()
	//	output, err := layers.BuildIO(fmt, dtype, dims)

	return &SoftMax{
		algo:  s.Flgs.Algo.Fast(),
		mode:  s.Flgs.Mode.Instance(),
		alpha: gocudnn.CFloat(1),
		beta:  gocudnn.CFloat(0),
	}
}

//ForwardProp performs the forward propigation
func (s *SoftMax) ForwardProp(handle *gocudnn.Handle, x, y *tensor.Tensor) error {

	err := s.helper.Funcs.SoftMaxForward(handle, s.algo, s.mode, s.alpha, x.TD(), x.Memer(), s.beta, y.TD(), y.Memer())
	return err
}

//BackProp performs the backward propigation
func (s *SoftMax) BackProp(handle *gocudnn.Handle, y, dy, dx *tensor.Tensor) error {

	err := s.helper.Funcs.SoftMaxBackward(handle, s.algo, s.mode, s.alpha, y.TD(), y.Memer(), dy.TD(), dy.Memer(), s.beta, dx.TD(), dx.Memer())
	return err
}
