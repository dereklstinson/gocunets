package gocunets

import (
	"errors"
	"strings"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
)

func (l *layer) loadparams(handle *cudnn.Handler, params *Params) error {
	var err error
	if l.cnn != nil {
		params.Layer = strings.ToUpper(params.Layer)
		if params.Layer != "CNN" {
			return errors.New("not a " + params.Layer + " layer")
		}
		err = params.Weight.LoadTensor(handle, l.cnn.Weights().T())
		if err != nil {
			return err
		}
		return params.Bias.LoadTensor(handle, l.cnn.Bias().T())

	}
	if l.batch != nil {
		params.Layer = strings.ToUpper(params.Layer)
		if params.Layer != "BATCH" {
			return errors.New("not a " + params.Layer + " layer")
		}
		err = params.Weight.LoadTensor(handle, l.batch.Scale().T())
		if err != nil {
			return err
		}
		return params.Bias.LoadTensor(handle, l.batch.Bias().T())
	}
	if l.cnntranspose != nil {

		params.Layer = strings.ToUpper(params.Layer)
		if params.Layer != "CNNTRANSPOSE" {
			return errors.New("not a " + params.Layer + " layer")
		}
		err = params.Weight.LoadTensor(handle, l.cnntranspose.Weights().T())
		if err != nil {
			return err
		}
		return params.Bias.LoadTensor(handle, l.cnntranspose.Bias().T())
	}
	if l.activation != nil {

		params.Layer = strings.ToUpper(params.Layer)
		if params.Layer != "ACTIVATION" {
			return errors.New("not a " + params.Layer + " layer")
		}
		if l.activation.NegCoefs() == nil {
			return errors.New("Activation doesn't have weights")
		}
		err = params.Weight.LoadTensor(handle, l.activation.NegCoefs().T())
		if err != nil {
			return err
		}
		if l.activation.PosCoefs() == nil {
			return nil
		}
		err = params.Bias.LoadTensor(handle, l.activation.PosCoefs().T())
		if err != nil {
			return err
		}
		err = params.Xtra.LoadTensor(handle, l.activation.Threshhold().T())
		if err != nil {
			return err
		}
	}

	return nil
}
func (l *layer) hasweights() bool {
	if l.cnn != nil {
		return true
	}
	if l.batch != nil {
		return true
	}
	if l.cnntranspose != nil {
		return true
	}
	if l.activation != nil {
		if l.activation.TrainersNeeded() > 0 {
			return true
		}

	}
	return false
}
func (l *layer) params() (*Params, error) {
	if l.cnn != nil {
		weights, err := getTensor(l.cnn.Weights().T())
		if err != nil {
			return nil, err
		}
		bias, err := getTensor(l.cnn.Bias().T())
		if err != nil {
			return nil, err
		}
		return &Params{
			Layer:  "CNN",
			Weight: weights,
			Bias:   bias,
		}, nil
	}
	if l.batch != nil {
		weights, err := getTensor(l.batch.Scale().T())
		if err != nil {
			return nil, err
		}
		bias, err := getTensor(l.batch.Bias().T())
		if err != nil {
			return nil, err
		}
		return &Params{
			Layer:  "BATCH",
			Weight: weights,
			Bias:   bias,
		}, nil
	}
	if l.cnntranspose != nil {
		weights, err := getTensor(l.cnntranspose.Weights().T())
		if err != nil {
			return nil, err
		}
		bias, err := getTensor(l.cnntranspose.Bias().T())
		if err != nil {
			return nil, err
		}
		return &Params{
			Layer:  "CNNTRANSPOSE",
			Weight: weights,
			Bias:   bias,
		}, nil
	}
	if l.activation != nil {
		if l.activation.TrainersNeeded() > 0 {

			negcoef, err := getTensor(l.activation.NegCoefs().T())
			if err != nil {
				return nil, err
			}
			if l.activation.PosCoefs().T() == nil {
				return &Params{
					Layer:  "ACTIVATION",
					Weight: negcoef,
				}, nil
			}
			thresh, err := getTensor(l.activation.Threshhold().T())
			if err != nil {
				return nil, err
			}
			poscoef, err := getTensor(l.activation.PosCoefs().T())
			if err != nil {
				return nil, err
			}
			return &Params{
				Layer:  "ACTIVATION",
				Weight: negcoef,
				Bias:   poscoef,
				Xtra:   thresh,
			}, nil
		}
	}
	return nil, nil
}
