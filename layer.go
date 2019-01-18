package gocunets

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/batchnorm"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/cnntranspose"
	"github.com/dereklstinson/GoCuNets/layers/dropout"
	"github.com/dereklstinson/GoCuNets/layers/fcnn"
	"github.com/dereklstinson/GoCuNets/layers/pooling"
	"github.com/dereklstinson/GoCuNets/layers/reshape"
	"github.com/dereklstinson/GoCuNets/layers/softmax"
	"github.com/dereklstinson/GoCuNets/trainer"
)

type layer struct {
	name         string
	activation   *activation.Layer
	cnn          *cnn.Layer
	fcnn         *fcnn.Layer
	softmax      *softmax.Layer
	pool         *pooling.Layer
	drop         *dropout.Layer
	batch        *batchnorm.Layer
	reshape      *reshape.Layer
	cnntranspose *cnntranspose.Layer
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
			Layer:  "Batch",
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
			Layer:  "CnnTranspose",
			Weight: weights,
			Bias:   bias,
		}, nil
	}
	if l.activation != nil {
		if l.activation.HasWeights() {

			negcoef, err := getTensor(l.activation.NegCoefs().T())
			if err != nil {
				return nil, err
			}
			if l.activation.PosCoefs().T() == nil {
				return &Params{
					Layer:  "Activation",
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
				Layer:  "Activation",
				Weight: negcoef,
				Bias:   poscoef,
				Xtra:   thresh,
			}, nil
		}
	}
	return nil, nil
}
func (l *layer) loadtrainer(handle *cudnn.Handler, trainerweights, trainerbias trainer.Trainer) error {
	if l.cnn != nil {
		return l.cnn.LoadTrainer(handle, trainerweights, trainerbias)
	}
	if l.fcnn != nil {
		return l.fcnn.LoadTrainer(handle, trainerweights, trainerbias)

	}
	if l.batch != nil {
		return l.batch.LoadTrainer(handle, trainerweights, trainerbias)
	}
	if l.cnntranspose != nil {
		return l.cnntranspose.LoadTrainer(handle, trainerweights, trainerbias)
	}
	return errors.New("inbedded error doesn't support trainers")
}
func (l *layer) needstrainer() bool {
	if l.cnn != nil {

		return true
	}
	if l.fcnn != nil {
		return true
	}
	if l.cnntranspose != nil {
		return true
	}
	if l.batch != nil {
		return true
	}
	return false
}

func wraplayer(input interface{}) (*layer, bool) { //the bool is for a counter to count the layers that contain weights
	switch l := input.(type) {

	case *activation.Layer:
		return &layer{
			activation: l,
			name:       "Activation",
		}, false
	case *cnn.Layer:
		return &layer{
			cnn:  l,
			name: "CNN",
		}, true
	case *fcnn.Layer:
		return &layer{
			fcnn: l,
			name: "FCNN",
		}, true
	case *softmax.Layer:
		return &layer{
			softmax: l,
			name:    "SoftMax",
		}, false
	case *pooling.Layer:
		return &layer{
			pool: l,
			name: "Pooling",
		}, false
	case *dropout.Layer:
		return &layer{
			drop: l,
			name: "DropOut",
		}, false
	case *batchnorm.Layer:
		return &layer{
			batch: l,
			name:  "BatchNorm",
		}, false
	case *reshape.Layer:
		return &layer{
			reshape: l,
			name:    "Reshape",
		}, false
	case *cnntranspose.Layer:
		return &layer{
			cnntranspose: l,
			name:         "CNN-Transpose",
		}, true

	default:
		return nil, false
	}
}

func (l *layer) getoutputwithname(handle *cudnn.Handler, input *layers.IO) (*layers.IO, string, error) {

	if l.cnn != nil {
		x, err := l.cnn.MakeOutputTensor(handle, input)
		return x, "CNN-Output", err
	}
	if l.fcnn != nil {
		_, _, dims, err := input.Properties()
		if err != nil {
			return nil, "", err
		}
		x, err := l.fcnn.MakeOutputTensor(int(dims[0]))
		return x, "FCNN-Output", err
	}
	if l.pool != nil {
		x, err := l.pool.MakeOutputLayer(input)
		return x, "Pooling-Output", err
	}
	if l.drop != nil {

		err := l.drop.BuildFromPreset(handle, input)
		if err != nil {

			return nil, "", err
		}
		x, err := input.ZeroClone()
		return x, "DropOut-Output", err
	}
	if l.activation != nil {
		x, err := input.ZeroClone()
		return x, "Activation-Output", err
	}
	if l.batch != nil {
		err := l.batch.SetupPreset(handle, input)
		if err != nil {
			return nil, "", err
		}
		x, err := input.ZeroClone()
		return x, "BatchNorm-Output", err
	}

	if l.softmax != nil {
		x, err := input.ZeroClone()
		return x, "SoftMax-Output", err
	}
	if l.reshape != nil {
		x, err := l.reshape.MakeOutputTensor(handle, input)
		return x, "Reshape-Output", err
	}
	if l.cnntranspose != nil {
		x, err := l.cnntranspose.MakeOutputTensor(handle, input)
		return x, "CnnTranspose-Output", err
	}
	return nil, "", errors.New("Layer Needs Support")
}
func (l *layer) getoutput(handle *cudnn.Handler, input *layers.IO) (*layers.IO, error) {

	if l.cnn != nil {
		return l.cnn.MakeOutputTensor(handle, input)
	}
	if l.fcnn != nil {
		_, _, dims, err := input.Properties()
		if err != nil {
			return nil, err
		}
		return l.fcnn.MakeOutputTensor(int(dims[0]))
	}
	if l.pool != nil {
		return l.pool.MakeOutputLayer(input)
	}
	if l.drop != nil {

		err := l.drop.BuildFromPreset(handle, input)
		if err != nil {

			return nil, err
		}
		return input.ZeroClone()
	}
	if l.activation != nil {

		return input.ZeroClone()
	}
	if l.batch != nil {

		err := l.batch.SetupPreset(handle, input)
		if err != nil {
			return nil, err
		}
		return input.ZeroClone()
	}

	if l.softmax != nil {
		return input.ZeroClone()
	}
	if l.reshape != nil {
		return l.reshape.MakeOutputTensor(handle, input)
	}
	if l.cnntranspose != nil {
		return l.cnntranspose.MakeOutputTensor(handle, input)
	}
	return nil, errors.New("Layer Needs Support")
}

//UpdateWeights updates the weights of layer
func (l *layer) updateWeights(handle *cudnn.Handler, batch int) error {
	var err error
	if l.cnn != nil {
		err = l.cnn.UpdateWeights(handle, batch)
	}
	if l.fcnn != nil {
		err = l.fcnn.UpdateWeights(handle, batch)
	}
	if l.cnntranspose != nil {
		err = l.cnntranspose.UpdateWeights(handle, batch)

	}
	if l.batch != nil {
		err = l.batch.UpdateWeights(handle, batch)
	}
	return err
}
func (l *layer) l1l2loss() (l1, l2 float32) {

	if l.cnn != nil {
		return l.cnn.L1L2Loss()
	}
	if l.fcnn != nil {
		return l.fcnn.L1L2Loss()
	}
	if l.cnntranspose != nil {
		return l.cnntranspose.L1L2Loss()

	}
	return -123, -123
}
