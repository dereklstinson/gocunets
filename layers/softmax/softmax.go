package softmax

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor/softmax"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCudnn"
)

//Layer is a layer that holds the algos for softmax
type Layer struct {
	s      *softmax.Ops
	alpha  float64
	beta   float64
	balpha float64
	bbeta  float64
}

//BuildDefault builds a default layer it takes an input and the output io and checks to make sure the format, dims, and datatype match up
func BuildDefault(input *layers.IO, answers *layers.IO) (*Layer, error) {
	fmt, dtype, dims, err := input.Properties()
	if err != nil {
		return nil, err
	}
	fmt1, dtype1, dims1, err := answers.Properties()
	if err != nil {
		return nil, err
	}
	if fmt1 != fmt {
		return nil, errors.New("input and answers tensors formats don't match")
	}
	if dtype != dtype1 {
		return nil, errors.New("input and answers tensor datatypes don't match")
	}
	if len(dims) != len(dims1) {
		return nil, errors.New("input and answers tensor dim lengths don't match")
	}
	for i := 0; i < len(dims); i++ {
		if dims[i] != dims1[i] {
			return nil, errors.New("input and answers tensor dims don't match")
		}
	}

	sftmax := softmax.DefaultOperation()
	if err != nil {
		return nil, err
	}
	return &Layer{
		s:      sftmax,
		alpha:  1.0,
		beta:   0.0,
		bbeta:  0.0,
		balpha: -1.0,
	}, nil
}

//ForwardProp performs the forward propigation y is the output
func (l *Layer) ForwardProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	return l.s.ForwardProp(handle, l.alpha, x.T(), l.beta, y.T())
	//	err := s.Funcs.SoftMaxForward(handle, l.algo, l.mode, l.alpha, x.T().TD(), x.T().Memer(), l.beta, y.T().TD(), y.T().Memer())
	//	return err
}

//BackProp performs the backward propigation // x is the output
func (l *Layer) BackProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	return l.s.BackProp(handle, l.balpha, y.T(), y.DeltaT(), l.bbeta, x.DeltaT())
	//	err := s.Funcs.SoftMaxBackward(handle, l.algo, l.mode, l.alpha, y.T().TD(), y.T().Memer(), y.T().TD(), y.DMem(), l.beta, x.DeltaT().TD(), x.DeltaT().Memer())
	//	return err
}

/*
func (l *Layer) LoadAnswer(y *layers.IO, answers gocudnn.Memer, flag gocudnn.MemcpyKind) error {
	dest := y.DeltaT().Memer().ByteSize()
	src := answers.ByteSize()
	if dest != src {
		return errors.New("Memory dest and src not equal")
	}
	return gocudnn.CudaMemCopy(y.DeltaT().Memer(), answers, dest, flag)

}
*/
