package softmax

import (
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCudnn"
)

type Layer struct {
	algo  gocudnn.SoftMaxAlgorithm
	mode  gocudnn.SoftMaxMode
	alpha gocudnn.CScalar
	beta  gocudnn.CScalar
}

func BuildDefault() *Layer {
	var s gocudnn.SoftMax

	return &Layer{
		algo:  s.Flgs.Algo.Fast(),
		mode:  s.Flgs.Mode.Instance(),
		alpha: gocudnn.CFloat(1),
		beta:  gocudnn.CFloat(0),
	}
}

func (l *Layer) ForwardProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	var s gocudnn.SoftMax
	err := s.Funcs.SoftMaxForward(handle, l.algo, l.mode, l.alpha, x.TensorD(), x.Mem(), l.beta, y.TensorD(), y.Mem())
	return err
}

func (l *Layer) BackProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	var s gocudnn.SoftMax
	err := s.Funcs.SoftMaxBackward(handle, l.algo, l.mode, l.alpha, y.TensorD(), y.Mem(), y.TensorD(), y.DMem(), l.beta, x.TensorD(), x.DMem())
	return err
}
