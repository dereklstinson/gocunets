package softmax

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	"github.com/dereklstinson/GoCudnn"
)

//Ops does the softmax algo
type Ops struct {
	algo      gocudnn.SoftMaxAlgorithm
	helper    gocudnn.SoftMax
	mode      gocudnn.SoftMaxMode
	alpha     gocudnn.CScalar
	beta      gocudnn.CScalar
	backalpha gocudnn.CScalar
	backbeta  gocudnn.CScalar
}

//Build builds a default layer (only option for now)
func Build() *Ops {
	var s gocudnn.SoftMax
	//	fmt, dtype, dims, err := input.Properties()
	//	output, err := layers.BuildIO(fmt, dtype, dims)

	return &Ops{
		algo:      s.Flgs.Algo.Fast(),
		mode:      s.Flgs.Mode.Channel(),
		backalpha: gocudnn.CFloat(-1),
		alpha:     gocudnn.CFloat(1),
		beta:      gocudnn.CFloat(0),
		backbeta:  gocudnn.CFloat(0),
	}
}

//ForwardProp performs the forward propigation
func (s *Ops) ForwardProp(handle *gocudnn.Handle, x, y *tensor.Volume) error {

	err := s.helper.Funcs.SoftMaxForward(handle, s.algo, s.mode, s.alpha, x.TD(), x.Memer(), s.beta, y.TD(), y.Memer())
	return err
}

//BackProp performs the backward propigation
func (s *Ops) BackProp(handle *gocudnn.Handle, y, dy, dx *tensor.Volume) error {

	err := s.helper.Funcs.SoftMaxBackward(handle, s.algo, s.mode, s.backalpha, y.TD(), y.Memer(), dy.TD(), dy.Memer(), s.backbeta, dx.TD(), dx.Memer())
	return err
}
