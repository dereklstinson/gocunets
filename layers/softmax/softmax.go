package softmax

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/gocudnn/softmax"
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

//Settings contains all the info that is needed in order to perform the softmaxoutput
type Settings struct {
	Mode gocudnn.SoftMaxMode      `json:"mode,omitempty"`
	Algo gocudnn.SoftMaxAlgorithm `json:"algo,omitempty"`
}

//BuildNormal will take the mode and algo flags and build it accordingly
func BuildNormal(input *layers.IO, answers *layers.IO, mode gocudnn.SoftMaxMode, algo gocudnn.SoftMaxAlgorithm) (*Layer, error) {
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

	sftmax := softmax.StageOperation(algo, mode)
	if err != nil {
		return nil, err
	}
	return &Layer{
		s:      sftmax,
		alpha:  defaultalpha,
		beta:   defaultbeta,
		bbeta:  defaultbeta,
		balpha: defaultbalpha,
	}, nil

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
		alpha:  defaultalpha,
		beta:   defaultbeta,
		bbeta:  defaultbeta,
		balpha: defaultbalpha,
	}, nil
}

const defaultalpha = 1.0
const defaultbeta = 0.0

const defaultbalpha = -1.0

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
