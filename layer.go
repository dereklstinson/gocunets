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

func (l *layer) loadtrainer(handle *cudnn.Handler, trainerweights, trainerbias trainer.Trainer) error {
	if l.cnn != nil {
		return l.cnn.LoadTrainer(handle, trainerweights, trainerbias)
	}
	if l.fcnn != nil {
		return l.fcnn.LoadTrainer(handle, trainerweights, trainerbias)

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
	return false
}

func wraplayer(input interface{}) (*layer, bool) { //the bool is for a counter to count the layers that contain weights
	switch l := input.(type) {

	case *activation.Layer:
		return &layer{
			activation: l,
		}, false
	case *cnn.Layer:
		return &layer{
			cnn: l,
		}, true
	case *fcnn.Layer:
		return &layer{
			fcnn: l,
		}, true
	case *softmax.Layer:
		return &layer{
			softmax: l,
		}, false
	case *pooling.Layer:
		return &layer{
			pool: l,
		}, false
	case *dropout.Layer:
		return &layer{
			drop: l,
		}, false
	case *batchnorm.Layer:
		return &layer{
			batch: l,
		}, false
	case *reshape.Layer:
		return &layer{
			reshape: l,
		}, false
	case *cnntranspose.Layer:
		return &layer{
			cnntranspose: l,
		}, true

	default:
		return nil, false
	}
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
		return input.ZeroClone()
	}
	if l.activation != nil {

		return input.ZeroClone()
	}
	if l.batch != nil {
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
