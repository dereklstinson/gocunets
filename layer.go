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
	gocudnn "github.com/dereklstinson/GoCudnn"
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

func wraplayer(input interface{}) *layer {
	switch l := input.(type) {

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
	case *cnntranspose.Layer:
		return &layer{
			cnntranspose: l,
		}

	default:
		return nil
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
		err = handle.Sync()
		if err != nil {
			return err
		}
		err = l.cnn.UpdateWeights(handle, batch)
		err = handle.Sync()
		if err != nil {
			return err
		}
		return err
	}
	if l.fcnn != nil {
		err = handle.Sync()
		if err != nil {
			return err
		}
		err = l.fcnn.UpdateWeights(handle, batch)
		if err != nil {
			return err
		}

		return handle.Sync()
	}
	if l.cnntranspose != nil {
		err = handle.Sync()
		if err != nil {
			return err
		}
		err = l.cnntranspose.UpdateWeights(handle, batch)
		if err != nil {
			return err
		}

		return handle.Sync()
	}
	return nil
}

//ForwardProp does the forward prop for a layer
func (l *layer) forwardprop(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {

	err := handle.Sync()
	if err != nil {
		return err
	}
	if l.cnn != nil {
		return l.cnn.ForwardProp(handle, wspace, x, y)
	}
	if l.fcnn != nil {
		return l.fcnn.ForwardProp(handle, x, y)
	}
	if l.drop != nil {
		return l.drop.ForwardProp(handle, x, y)
	}
	if l.activation != nil {

		return l.activation.ForwardProp(handle, x, y)
	}
	if l.softmax != nil {
		return l.softmax.ForwardProp(handle, x, y)
	}
	if l.pool != nil {
		return l.pool.ForwardProp(handle, x, y)
	}

	if l.reshape != nil {
		err = l.reshape.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()

	}
	if l.batch != nil {
		l.batch.ForwardProp(handle, x, y)
	}
	if l.cnntranspose != nil {
		err = l.cnntranspose.ForwardProp(handle, wspace, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	return errors.New("Layer Not Set Up")
}

//BackProp does the backprop of a layer
func (l *layer) backpropfilterdata(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	err := handle.Sync()
	if err != nil {
		return err
	}
	if l.cnn != nil {
		err = l.cnn.BackPropFilterData(handle, wspace, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()

	}

	if l.fcnn != nil {
		err = l.fcnn.BackPropFilterData(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}

	if l.activation != nil {
		err = l.activation.BackProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}

	if l.softmax != nil {
		err = l.softmax.BackProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}

	if l.drop != nil {
		err = l.drop.BackProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}

	if l.pool != nil {
		err = l.pool.BackProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}

	if l.reshape != nil {
		err = l.reshape.BackProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.batch != nil {
		err = l.batch.BackProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.cnntranspose != nil {
		err = l.cnntranspose.BackPropFilterData(handle, wspace, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	return errors.New("Layer Not Set Up")
}

//BackProp does the backprop of a layer
func (l *layer) backpropdata(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	err := handle.Sync()
	if err != nil {
		return err
	}
	if l.cnn != nil {
		err = l.cnn.BackPropData(handle, wspace, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.fcnn != nil {
		err = l.fcnn.BackPropData(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.activation != nil {
		err = l.activation.BackProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.softmax != nil {
		err = l.softmax.BackProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.drop != nil {
		err = l.drop.BackProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.pool != nil {
		err = l.pool.BackProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.reshape != nil {
		err = l.reshape.BackProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.batch != nil {
		err = l.batch.BackProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.cnntranspose != nil {
		err = l.cnntranspose.BackPropData(handle, wspace, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	return errors.New("Layer Not Set Up")
}
