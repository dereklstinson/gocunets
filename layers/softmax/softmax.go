package softmax

import (
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCudnn"
)

//Layer is a layer that holds the algos for softmax
type Layer struct {
	algo  gocudnn.SoftMaxAlgorithm
	mode  gocudnn.SoftMaxMode
	alpha gocudnn.CScalar
	beta  gocudnn.CScalar
}

//BuildDefault builds a default layer (only option for now)
func BuildDefault() *Layer {
	var s gocudnn.SoftMax

	return &Layer{
		algo:  s.Flgs.Algo.Fast(),
		mode:  s.Flgs.Mode.Instance(),
		alpha: gocudnn.CFloat(1),
		beta:  gocudnn.CFloat(0),
	}
}

//ForwardProp performs the forward propigation
func (l *Layer) ForwardProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	var s gocudnn.SoftMax
	err := s.Funcs.SoftMaxForward(handle, l.algo, l.mode, l.alpha, x.Tensor().TD(), x.Tensor().Memer(), l.beta, y.Tensor().TD(), y.Tensor().Memer())
	return err
}

//BackProp performs the backward propigation
func (l *Layer) BackProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	var s gocudnn.SoftMax
	err := s.Funcs.SoftMaxBackward(handle, l.algo, l.mode, l.alpha, y.Tensor().TD(), y.Tensor().Memer(), y.Tensor().TD(), y.DMem(), l.beta, x.DTensor().TD(), x.DTensor().Memer())
	return err
}
