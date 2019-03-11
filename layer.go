package gocunets

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/batchnorm"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/cnntranspose"
	"github.com/dereklstinson/GoCuNets/layers/dropout"
	"github.com/dereklstinson/GoCuNets/layers/pooling"
	"github.com/dereklstinson/GoCuNets/layers/reshape"
	"github.com/dereklstinson/GoCuNets/layers/softmax"
	"github.com/dereklstinson/GoCuNets/trainer"
)

type layer struct {
	name         string
	activation   *activation.Layer
	cnn          *cnn.Layer
	softmax      *softmax.Layer
	pool         *pooling.Layer
	drop         *dropout.Layer
	batch        *batchnorm.Layer
	reshape      *reshape.Layer
	cnntranspose *cnntranspose.Layer
	scalarnum    int
}

func (l *layer) loadtrainer(handle *cudnn.Handler, trainerweights, trainerbias trainer.Trainer) error {
	if l.cnn != nil {
		return l.cnn.LoadTrainer(handle, trainerweights, trainerbias)
	}

	if l.batch != nil {
		return l.batch.LoadTrainer(handle, trainerweights, trainerbias)
	}
	if l.cnntranspose != nil {
		return l.cnntranspose.LoadTrainer(handle, trainerweights, trainerbias)
	}
	if l.activation != nil {
		if l.activation.Updateable() {
			return l.activation.LoadTrainer(handle, trainerweights, trainerbias)
		}
	}
	return errors.New("inbedded error doesn't support trainers")
}

func (l *layer) needstrainer() bool {
	if l.cnn != nil {

		return true
	}

	if l.cnntranspose != nil {
		return true
	}
	if l.batch != nil {
		return true
	}
	if l.activation != nil {
		return l.activation.Updateable()
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
func (l *layer) numofscalars() int {
	return l.scalarnum
}
func (l *layer) initalphascalarsamount() int {

	if l.cnn != nil {
		l.scalarnum = l.cnn.NumAlphaScalars()
		return l.scalarnum
	}

	if l.pool != nil {
		l.scalarnum = l.pool.NumAlphaScalars()
		return l.scalarnum

	}
	if l.drop != nil {

		return 0
	}
	if l.activation != nil {
		l.scalarnum = l.activation.NumAlphaScalars()
		return l.scalarnum

	}
	if l.batch != nil {
		return 0
	}

	if l.softmax != nil {
		l.scalarnum = l.softmax.NumAlphaScalars()
		return l.scalarnum

	}
	if l.reshape != nil {
		return 0
	}
	if l.cnntranspose != nil {
		l.scalarnum = l.cnntranspose.NumAlphaScalars()
		return l.scalarnum

	}
	return 0

}
func (l *layer) initbetascalarsamount() int {

	if l.cnn != nil {
		l.scalarnum = l.cnn.NumBetaScalars()
		return l.scalarnum
	}

	if l.pool != nil {
		l.scalarnum = l.pool.NumBetaScalars()
		return l.scalarnum

	}
	if l.drop != nil {

		return 0
	}
	if l.activation != nil {
		l.scalarnum = l.activation.NumBetaScalars()
		return l.scalarnum

	}
	if l.batch != nil {
		return 0
	}

	if l.softmax != nil {
		l.scalarnum = l.softmax.NumBetaScalars()
		return l.scalarnum

	}
	if l.reshape != nil {
		return 0
	}
	if l.cnntranspose != nil {
		l.scalarnum = l.cnntranspose.NumBetaScalars()
		return l.scalarnum

	}
	return 0

}
func (l *layer) updateabetascalar(scalars []float64) (offset []float64) {
	if l.cnn != nil {

		l.cnn.SetBetaScalars(scalars[:l.scalarnum])
		return scalars[l.scalarnum:]
	}

	if l.pool != nil {
		l.pool.SetBetaScalars(scalars[:l.scalarnum])
		return scalars[l.scalarnum:]

	}
	if l.drop != nil {

		return scalars
	}
	if l.activation != nil {
		l.activation.SetBetaScalars(scalars[:l.scalarnum])
		return scalars[l.scalarnum:]

	}
	if l.batch != nil {
		return scalars
	}

	if l.softmax != nil {
		l.softmax.SetBetaScalars(scalars[:l.scalarnum])
		return scalars[l.scalarnum:]

	}
	if l.reshape != nil {
		return scalars
	}
	if l.cnntranspose != nil {
		l.cnntranspose.SetBetaScalars(scalars[:l.scalarnum])
		return scalars[l.scalarnum:]

	}
	return scalars
}
func (l *layer) updatealphascalar(scalars []float64) (offset []float64) {
	if l.cnn != nil {

		l.cnn.SetAlphaScalars(scalars[:l.scalarnum])
		return scalars[l.scalarnum:]
	}

	if l.pool != nil {
		l.pool.SetAlphaScalars(scalars[:l.scalarnum])
		return scalars[l.scalarnum:]

	}
	if l.drop != nil {

		return scalars
	}
	if l.activation != nil {
		l.activation.SetAlphaScalars(scalars[:l.scalarnum])
		return scalars[l.scalarnum:]

	}
	if l.batch != nil {
		return scalars
	}

	if l.softmax != nil {
		l.softmax.SetAlphaScalars(scalars[:l.scalarnum])
		return scalars[l.scalarnum:]

	}
	if l.reshape != nil {
		return scalars
	}
	if l.cnntranspose != nil {
		l.cnntranspose.SetAlphaScalars(scalars[:l.scalarnum])
		return scalars[l.scalarnum:]

	}
	return scalars
}
func (l *layer) getoutputwithname(handle *cudnn.Handler, input *layers.IO) (*layers.IO, string, error) {

	if l.cnn != nil {
		x, err := l.cnn.MakeOutputTensor(handle, input)
		return x, "CNN-Output", err
	}

	if l.pool != nil {
		x, err := l.pool.MakeOutputLayer(handle, input)
		return x, "Pooling-Output", err
	}
	if l.drop != nil {

		err := l.drop.BuildFromPreset(handle, input)
		if err != nil {

			return nil, "", err
		}
		x, err := input.ZeroClone(handle)
		return x, "DropOut-Output", err
	}
	if l.activation != nil {
		x, err := input.ZeroClone(handle)
		return x, "Activation-Output", err
	}
	if l.batch != nil {
		err := l.batch.SetupPreset(handle, input)
		if err != nil {
			return nil, "", err
		}
		x, err := input.ZeroClone(handle)
		return x, "BatchNorm-Output", err
	}

	if l.softmax != nil {
		x, err := input.ZeroClone(handle)
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

/*
func (l *layer) getoutputdims(handle *cudnn.Handler, input *layers.IO) ([]int32, error) {

}
*/
func (l *layer) getoutput(handle *cudnn.Handler, input *layers.IO) (*layers.IO, error) {

	if l.cnn != nil {
		io, err := l.cnn.MakeOutputTensor(handle, input)
		if err != nil {
			fmt.Println("Error in CNN Make Output Tensor input is:", input)
		}
		return io, err
	}

	if l.pool != nil {
		return l.pool.MakeOutputLayer(handle, input)
	}
	if l.drop != nil {

		err := l.drop.BuildFromPreset(handle, input)
		if err != nil {

			return nil, err
		}
		return input.ZeroClone(handle)
	}
	if l.activation != nil {
		io, err := input.ZeroClone(handle)
		if err != nil {
			fmt.Println("Error in activation Make Output Tensor input is:", input)
		}
		return io, err

	}
	if l.batch != nil {

		err := l.batch.SetupPreset(handle, input)
		if err != nil {
			fmt.Println("error in batch initialization")
			return nil, err
		}

		io, err := input.ZeroClone(handle)
		if err != nil {
			fmt.Println("Error in batch Make Output Tensor input is:", input)
		}
		return io, err
	}

	if l.softmax != nil {
		return input.ZeroClone(handle)
	}
	if l.reshape != nil {
		return l.reshape.MakeOutputTensor(handle, input)
	}
	if l.cnntranspose != nil {
		io, err := l.cnntranspose.MakeOutputTensor(handle, input)
		if err != nil {
			fmt.Println("Error in cnntranspose Make Output Tensor input is:", input)
		}
		return io, err
	}
	return nil, errors.New("Layer Needs Support")
}

//UpdateWeights updates the weights of layer
func (l *layer) updateWeights(handle *cudnn.Handler, batch int) error {

	if l.cnn != nil {
		return l.cnn.UpdateWeights(handle, batch)
	}

	if l.cnntranspose != nil {
		return l.cnntranspose.UpdateWeights(handle, batch)

	}
	if l.batch != nil {
		return l.batch.UpdateWeights(handle, batch)
	}
	if l.activation != nil {
		if l.activation.Updateable() {
			return l.activation.UpdateWeights(handle, batch)

		}

	}
	return nil
}

func (l *layer) l1l2loss() (l1, l2 float32) {

	if l.cnn != nil {
		return l.cnn.L1L2Loss()
	}

	if l.cnntranspose != nil {
		return l.cnntranspose.L1L2Loss()

	}
	return -123, -123
}
