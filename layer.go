package gocunets

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/batchnorm"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/dropout"
	"github.com/dereklstinson/GoCuNets/layers/fcnn"
	"github.com/dereklstinson/GoCuNets/layers/pooling"
	"github.com/dereklstinson/GoCuNets/layers/reshape"
	"github.com/dereklstinson/GoCuNets/layers/softmax"
	"github.com/dereklstinson/GoCuNets/layers/xactivation"
	"github.com/dereklstinson/GoCuNets/trainer"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

type layer struct {
	xactivation *xactivation.Layer
	activation  *activation.Layer
	cnn         *cnn.Layer
	fcnn        *fcnn.Layer
	softmax     *softmax.Layer
	pool        *pooling.Layer
	drop        *dropout.Layer
	batch       *batchnorm.Layer
	reshape     *reshape.Layer
}

//asdfas
func (l *layer) loadtrainer(handle *Handles, trainerweights, trainerbias trainer.Trainer) error {
	if l.cnn != nil {
		return l.cnn.LoadTrainer(handle.xhandle, trainerweights, trainerbias)
	}
	if l.fcnn != nil {
		return l.fcnn.LoadTrainer(handle.xhandle, trainerweights, trainerbias)

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
	return false
}

func wraplayer(input interface{}) *layer {
	switch l := input.(type) {
	case *xactivation.Layer:
		return &layer{
			xactivation: l,
		}
	case *activation.Layer:
		return &layer{
			activation: l,
		}
	case *cnn.Layer:
		return &layer{
			cnn: l,
		}
	case *fcnn.Layer:
		return &layer{
			fcnn: l,
		}
	case *softmax.Layer:
		return &layer{
			softmax: l,
		}
	case *pooling.Layer:
		return &layer{
			pool: l,
		}
	case *dropout.Layer:
		return &layer{
			drop: l,
		}
	case *batchnorm.Layer:
		return &layer{
			batch: l,
		}
	case *reshape.Layer:
		return &layer{
			reshape: l,
		}

	default:
		return nil
	}
}
func (l *layer) getoutput(handle *Handles, input *layers.IO) (*layers.IO, error) {

	if l.cnn != nil {
		return l.cnn.MakeOutputTensor(handle.cudnn, input)
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
	if l.xactivation != nil {
		return input.ZeroClone()
	}
	if l.softmax != nil {
		return input.ZeroClone()
	}
	if l.reshape != nil {
		return l.reshape.MakeOutputTensor(handle.xhandle, input)
	}

	return nil, errors.New("Layer Needs Support")
}

//UpdateWeights updates the weights of layer
func (l *layer) updateWeights(handle *Handles, batch int) error {
	if l.cnn != nil {
		return l.cnn.UpdateWeights(handle.xhandle, batch)
	}
	if l.fcnn != nil {
		return l.fcnn.UpdateWeights(handle.xhandle, batch)
	}
	return nil
}

//ForwardProp does the forward prop for a layer
func (l *layer) forwardprop(handle *Handles, wpace gocudnn.Memer, x, y *layers.IO) error {
	if l.cnn != nil {
		return l.cnn.ForwardProp(handle.cudnn, wpace, x, y)
	}
	if l.fcnn != nil {
		return l.fcnn.ForwardProp(handle.cudnn, x, y)
	}
	if l.drop != nil {
		return l.drop.ForwardProp(handle.cudnn, x, y)
	}
	if l.activation != nil {
		return l.activation.ForwardProp(handle.cudnn, x, y)
	}
	if l.softmax != nil {
		return l.softmax.ForwardProp(handle.cudnn, x, y)
	}
	if l.pool != nil {
		return l.pool.ForwardProp(handle.cudnn, x, y)
	}
	if l.xactivation != nil {
		return l.xactivation.ForwardProp(handle.xhandle, x, y)
	}
	if l.reshape != nil {
		return l.reshape.ForwardProp(handle.xhandle, x, y)
	}
	if l.batch != nil {
		l.batch.ForwardProp(handle.cudnn, x, y)
	}

	return errors.New("Layer Not Set Up")
}

//BackProp does the backprop of a layer
func (l *layer) backprop(handle *Handles, wpace gocudnn.Memer, x, y *layers.IO) error {
	if l.cnn != nil {
		return l.cnn.BackProp(handle.cudnn, wpace, x, y)
	}
	if l.fcnn != nil {
		return l.fcnn.BackProp(handle.cudnn, x, y)
	}
	if l.activation != nil {
		return l.activation.BackProp(handle.cudnn, x, y)
	}
	if l.softmax != nil {
		return l.softmax.BackProp(handle.cudnn, x, y)
	}
	if l.drop != nil {
		return l.drop.BackProp(handle.cudnn, x, y)
	}
	if l.pool != nil {
		return l.pool.BackProp(handle.cudnn, x, y)
	}
	if l.xactivation != nil {
		return l.xactivation.BackProp(handle.xhandle, x, y)
	}
	if l.reshape != nil {
		return l.reshape.BackProp(handle.xhandle, x, y)
	}
	if l.batch != nil {
		return l.batch.BackProp(handle.cudnn, x, y)
	}
	return errors.New("Layer Not Set Up")
}
