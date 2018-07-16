package softmax

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor/softmax"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCudnn"
)

//Layer is a layer that holds the algos for softmax
type Layer struct {
	s *softmax.SoftMax
}

//BuildDefault builds a default layer (only option for now)
func BuildDefault(input *layers.IO) (*Layer, *layers.IO, error) {

	fmt, dtype, dims, err := input.Properties()
	output, err := layers.BuildIO(fmt, dtype, dims)
	sftmax := softmax.BuildDefault()
	if err != nil {
		return nil, nil, err
	}
	return &Layer{
		s: sftmax,
	}, output, nil
}

//ForwardProp performs the forward propigation y is the output
func (l *Layer) ForwardProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	return l.s.ForwardProp(handle, x.T(), y.T())
	//	err := s.Funcs.SoftMaxForward(handle, l.algo, l.mode, l.alpha, x.T().TD(), x.T().Memer(), l.beta, y.T().TD(), y.T().Memer())
	//	return err
}

//BackProp performs the backward propigation // x is the output
func (l *Layer) BackProp(handle *gocudnn.Handle, y, x *layers.IO) error {
	return l.s.BackProp(handle, y.T(), y.DeltaT(), x.DeltaT())
	//	err := s.Funcs.SoftMaxBackward(handle, l.algo, l.mode, l.alpha, y.T().TD(), y.T().Memer(), y.T().TD(), y.DMem(), l.beta, x.DeltaT().TD(), x.DeltaT().Memer())
	//	return err
}
func (l *Layer) LoadAnswer(y *layers.IO, answers ffgocudnn.Memer, flag gocudnn.MemcpyKind) error {
	dest := y.DeltaT().Memer().ByteSize()
	src := answers.ByteSize()
	if dest != src {
		return errors.New("Memory dest and src not equal")
	}
	return gocudnn.CudaMemCopy(y.DeltaT().Memer(), answers, dest, flag)

}
