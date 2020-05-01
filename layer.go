package gocunets

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/layers"
	"github.com/dereklstinson/gocunets/layers/activation"
	"github.com/dereklstinson/gocunets/layers/batchnorm"
	"github.com/dereklstinson/gocunets/layers/cnn"
	"github.com/dereklstinson/gocunets/layers/cnntranspose"
	"github.com/dereklstinson/gocunets/layers/dropout"
	"github.com/dereklstinson/gocunets/layers/pooling"
	"github.com/dereklstinson/gocunets/layers/reshape"
	"github.com/dereklstinson/gocunets/trainer"
	"github.com/dereklstinson/gocudnn/gocu"
)

//Operation is a generic operation that a layer uses.
//
//The forward and backward don't need to use all the x,dx,y,and dy, but they do need to be passed.
//
type Operation interface {
	Forward(handle *cudnn.Handler, x, dx, y, dy *layers.Tensor) error
	Inference(handle *cudnn.Handler, x, y *layers.Tensor) error
	Backward(handle *cudnn.Handler, x, dx, y, dy *Tensor) error
	UpdateWeights(handle *cudnn.Handler) error
	LoadTrainers(handle *cudnn.Handler, trainers ...trainer.Trainer) error
	TrainersNeeded() int
	SetOtherScalars(alpha, beta float64)
	SetForwardScalars(alpha, beta float64)
	SetBackwardScalars(alpha, beta float64)
	GetOutputDims(input *layers.Tensor) ([]int32, error)
}

//Layer is a layer inside a network it holds inputs and outputs
type Layer struct {
	id              int64
	name            string
	h               *Handle
	activation      *activation.Layer
	cnn             *cnn.Layer
	pool            *pooling.Layer
	drop            *dropout.Layer
	batch           *batchnorm.Layer
	reshape         *reshape.Layer
	cnntranspose    *cnntranspose.Layer
	other           Operation //Operation will eventually take over
	x, dx, y, dy    *Tensor
	memoryeffecient bool

	s                                        gocu.Streamer
	workspacefwd, workspacebwd, workspacebwf *nvidia.Malloced
	batchsize                                int
	//scalarnumalpha, scalarnumbeta            int
}

//ToggleWPrint If layer contains weights or hidden values it will toggle the printing
func (l *Layer) ToggleWPrint() {
	if l.cnn != nil {
		l.cnn.ToggleWeightsPrintValueForStringer()
		return
	}
	if l.cnntranspose != nil {
		l.cnntranspose.ToggleWeightsPrintValueForStringer()
		return
	}
	return
}

//ToggleDWPrint If layer contains delta weights
func (l *Layer) ToggleDWPrint() {
	if l.cnn != nil {
		l.cnn.ToggleDWeightsPrintValueForStringer()
		return
	}
	if l.cnntranspose != nil {
		l.cnntranspose.ToggleDWeightsPrintValueForStringer()
		return
	}
	return
}
func (l *Layer) String() string {
	if l.cnn != nil {
		return l.cnn.String()
	}
	if l.cnntranspose != nil {
		return l.cnntranspose.String()
	}
	if l.activation != nil {
		return "NO Activation Stringer Yet"
	}
	if l.drop == nil {
		return "No Dropout stringer yet"
	}
	if l.reshape == nil {
		return "No Reshape stringer yet"
	}
	return "Unsupported Stringer for now"
}

//ToggleBiasPrint If layer contains bias.
func (l *Layer) ToggleBiasPrint() {
	if l.cnn != nil {
		l.cnn.ToggleBiasPrintValueForStringer()
		return
	}
	if l.cnntranspose != nil {
		l.cnntranspose.ToggleBiasPrintValueForStringer()
		return
	}
	return
}

//ToggleDBiasPrint If layer contains dbias
func (l *Layer) ToggleDBiasPrint() {
	if l.cnn != nil {
		l.cnn.ToggleDBiasPrintValueForStringer()
		return
	}
	if l.cnntranspose != nil {
		l.cnntranspose.ToggleDBiasPrintValueForStringer()
		return
	}
	return
}

//ToggleAllHiddenLayerValues toggles all hidden values on
func (l *Layer) ToggleAllHiddenLayerValues() {
	l.ToggleBiasPrint()
	l.ToggleDBiasPrint()
	l.ToggleDWPrint()
	l.ToggleWPrint()
}

//CreateOperationLayer creates an operation layer
func CreateOperationLayer(id int64, handle *Handle, op Operation) (l *Layer, err error) {
	l, err = createlayer(id, handle, op)
	return l, err
}

//CreateLayer creates a layer with generic
func createlayer(id int64, handle *Handle, op interface{}) (l *Layer, err error) {
	l = new(Layer)
	switch x := op.(type) {
	case *activation.Layer:
		l.activation = x
	case *cnn.Layer:
		l.cnn = x
	case *pooling.Layer:
		l.pool = x
	case *dropout.Layer:
		l.drop = x
	case *batchnorm.Layer:
		l.batch = x
	case *reshape.Layer:
		l.reshape = x
	case *cnntranspose.Layer:
		l.cnntranspose = x
	case Operation:
		l.other = x
	default:
		return nil, errors.New("Unsupported Layer")

	}
	l.h = handle
	return l, err

}

//ID is the ID of the layer
func (l *Layer) ID() int64 {
	return l.id
}

//SetIOs sets the x,dx,y,dy used by the layer
func (l *Layer) SetIOs(x, dx, y, dy *Tensor) {
	l.x, l.dx, l.y, l.dy = x, dx, y, dy
}

//SetInputs sets the inputs
func (l *Layer) SetInputs(x, dx *Tensor) {
	l.x, l.dx = x, dx
}

//SetOutputs sets the outputs
func (l *Layer) SetOutputs(y, dy *Tensor) {
	l.y, l.dy = y, dy
}

//Forward performs the forward propagation
func (l *Layer) Forward() error {
	return l.forwardprop()
}

//Backward performs the backward propagation
func (l *Layer) Backward() error {
	return l.backpropfilterdata()
}

//Update updates weights if layer has them
func (l *Layer) Update(epoch int) error {
	return l.updateWeights(epoch)
}

//ChangeBatchSize will change the batch size
func (l *Layer) ChangeBatchSize(batchsize int) {
	l.batchsize = batchsize
}

//LoadTrainer Loas the trainer to the layer
func (l *Layer) LoadTrainer(handle *cudnn.Handler, batchsize int, trainers ...trainer.Trainer) error {

	return l.loadtrainer(handle, batchsize, trainers...)
}
func (l *Layer) loadtrainer(handle *cudnn.Handler, batchsize int, trainers ...trainer.Trainer) error {
	l.batchsize = batchsize
	if l.cnn != nil {
		if len(trainers) != 2 {
			fmt.Println(len(trainers))
			return fmt.Errorf("l.cnn got %d should get %d", len(trainers), 2)
		}
		return l.cnn.LoadTrainer(handle, trainers[0], trainers[1])
	}

	if l.batch != nil {
		if len(trainers) != 2 {
			return fmt.Errorf("l.batch got %d should get %d", len(trainers), 2)
		}
		return l.batch.LoadTrainer(handle, trainers[0], trainers[1])
	}
	if l.cnntranspose != nil {
		if len(trainers) != 2 {

			return fmt.Errorf("l.cnntranspose got %d should get %d", len(trainers), 2)

		}
		return l.cnntranspose.LoadTrainer(handle, trainers[0], trainers[1])
	}
	if l.activation != nil {
		tneed := l.activation.TrainersNeeded()
		if tneed > 0 {

			if len(trainers) != tneed {

				return fmt.Errorf("l.activation got %d should get %d", len(trainers), tneed)
			}
		}
		return l.activation.LoadTrainer(handle, trainers)
	}
	if l.other != nil {
		tneed := l.other.TrainersNeeded()
		if tneed > 0 {

			if len(trainers) != tneed {

				return fmt.Errorf("l.other got %d should get %d", len(trainers), tneed)
			}
		}
		l.other.LoadTrainers(handle, trainers...)
	}

	return errors.New("inbedded error doesn't support trainers")
}

func (l *Layer) trainersneeded() int {
	if l.cnn != nil {
		return 2
	}
	if l.cnntranspose != nil {
		return 2
	}
	if l.batch != nil {
		return 2
	}
	if l.activation != nil {
		return l.activation.TrainersNeeded()

	}
	if l.activation != nil {
		return l.activation.TrainersNeeded()

	}
	if l.other != nil {
		return l.other.TrainersNeeded()
	}
	return 0

}

func wraplayer(input interface{}) (hidden *Layer, ios int) {
	switch l := input.(type) {

	case *activation.Layer:
		if l.TrainersNeeded() > 0 {
			return &Layer{
				activation: l,
				name:       "Activation",
			}, 1 + l.TrainersNeeded()
		}
		return &Layer{
			activation: l,
			name:       "Activation",
		}, 1

	case *cnn.Layer:
		return &Layer{
			cnn:  l,
			name: "CNN",
		}, 2

	case *pooling.Layer:
		return &Layer{
			pool: l,
			name: "Pooling",
		}, 1
	case *dropout.Layer:
		return &Layer{
			drop: l,
			name: "DropOut",
		}, 1
	case *batchnorm.Layer:
		return &Layer{
			batch: l,
			name:  "BatchNorm",
		}, 1
	case *reshape.Layer:
		return &Layer{
			reshape: l,
			name:    "Reshape",
		}, 1
	case *cnntranspose.Layer:
		return &Layer{
			cnntranspose: l,
			name:         "CNN-Transpose",
		}, 2

	default:
		return nil, -1
	}
}

//SetForwardScalars sets the forward scalars.
func (l *Layer) SetForwardScalars(alpha, beta float64) {
	if l.cnn != nil {
		l.cnn.SetForwardScalars(alpha, beta)
	} else if l.cnntranspose != nil {
		l.cnntranspose.SetForwardScalars(alpha, beta)
	} else if l.pool != nil {
		l.pool.SetForwardScalars(alpha, beta)

	} else if l.other != nil {
		l.other.SetForwardScalars(alpha, beta)
	}
	return
}

//SetBackwardScalars sets backward scalars
func (l *Layer) SetBackwardScalars(alpha, beta float64) {
	if l.cnn != nil {
		l.cnn.SetBackwardScalars(alpha, beta)
	} else if l.cnntranspose != nil {
		l.cnntranspose.SetBackwardScalars(alpha, beta)
	} else if l.pool != nil {
		l.pool.SetBackwardScalars(alpha, beta)

	} else if l.other != nil {
		l.other.SetBackwardScalars(alpha, beta)
	}
	return
}

//SetOtherScalars sets other scalars that the layer might have scalars
func (l *Layer) SetOtherScalars(alpha, beta float64) {
	if l.cnn != nil {
		l.cnn.SetOtherScalars(alpha, beta)
	} else if l.cnntranspose != nil {
		l.cnntranspose.SetOtherScalars(alpha, beta)
	} else if l.other != nil {
		l.other.SetOtherScalars(alpha, beta)
	}
	return
}

/*
func (l *Layer) getoutputwithname(handle *cudnn.Handler, input *layers.Tensor) (*layers.Tensor, string, error) {

	if l.cnn != nil {
		x, err := l.cnn.MakeOutputTensor(handle, input)
		return x, "CNN-Output", err
	}

	if l.pool != nil {
		x, err := l.pool.MakeOutputTensor(handle, input)
		return x, "Pooling-Output", err
	}
	if l.drop != nil {

		err := l.drop.BuildFromPreset(handle, input)
		if err != nil {

			return nil, "", err
		}
		x, err := layers.ZeroClone(handle, input)
		return x, "DropOut-Output", err
	}
	if l.activation != nil {
		x, err := layers.ZeroClone(handle, input)
		return x, "Activation-Output", err
	}
	if l.batch != nil {
		err := l.batch.SetupPreset(handle, input)
		if err != nil {
			return nil, "", err
		}
		x, err := layers.ZeroClone(handle, input)
		return x, "BatchNorm-Output", err
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
*/

//GetOutputDims gets the dims of the output tensor
func (l *Layer) GetOutputDims(input *Tensor) (output []int32, err error) {
	if l.cnn != nil {
		return l.cnn.FindOutputDims(input.Tensor)
	}
	if l.pool != nil {
		return l.pool.GetOutputDims(input.Tensor)
	}
	if l.batch != nil {
		output = make([]int32, len(input.Dims()))
		copy(output, input.Dims())
		return output, nil
	}
	if l.cnntranspose != nil {
		return l.cnntranspose.FindOutputDims(input.Tensor)
	}
	if l.activation != nil {
		output = make([]int32, len(input.Dims()))
		copy(output, input.Dims())
		return output, nil
	}
	if l.drop != nil {
		output = make([]int32, len(input.Dims()))
		copy(output, input.Dims())
		return output, nil
	}
	if l.other != nil {
		return l.other.GetOutputDims(input.Tensor)
	}
	return nil, errors.New("Unsupported Layer")
}

/*
func (l *Layer) getoutput(handle *cudnn.Handler, input *layers.Tensor) (io *layers.Tensor, err error) {

	if l.cnn != nil {
		io, err = l.cnn.MakeOutputTensor(handle, input)
		if io == nil {
			fmt.Println("input is", input.Dims())

		}
		if err != nil {
			fmt.Println("Error in CNN Make Output Tensor input is:", input)
		}
		return io, err
	}
	if l.pool != nil {
		io, err = l.pool.MakeOutputLayer(handle, input)
		if io == nil {
			panic("IO IS NILL")
		}
		return io, err
	}
	if l.drop != nil {
		err = l.drop.BuildFromPreset(handle, input)
		if err != nil {
			return nil, err
		}
		io, err = layers.ZeroClone(handle, input)
		if io == nil {
			panic("IO IS NILL")
		}
		return io, err

	}
	if l.activation != nil {
		io, err = layers.ZeroClone(handle, input)
		if err != nil {
			fmt.Println("Error in activation Make Output Tensor input is:", input)
		}
		if io == nil {
			panic("IO IS NILL")
		}

		return io, err
	}
	if l.batch != nil {
		err := l.batch.SetupPreset(handle, input)
		if err != nil {
			fmt.Println("error in batch initialization")
			return nil, err
		}

		io, err = layers.ZeroClone(handle, input)
		if err != nil {
			fmt.Println("Error in batch Make Output Tensor input is:", input)
		}
		if io == nil {
			panic("IO IS NILL")
		}

		return io, err
	}

	if l.reshape != nil {
		io, err = l.reshape.MakeOutputTensor(handle, input)
		if io == nil {
			panic("IO IS NILL")
		}
		return io, err
	}
	if l.cnntranspose != nil {
		io, err = l.cnntranspose.MakeOutputTensor(handle, input)
		if err != nil {
			fmt.Println("DIMS Reverse", io.Dims())
			fmt.Println("Error in cnntranspose Make Output Tensor input is:", input)
		}
		if io == nil {
			panic("IO IS NILL")
		}
		return io, err
	}
	return nil, errors.New("Layer Needs Support")
}
*/
//UpdateWeights updates the weights of layer
func (l *Layer) updateWeights(epoch int) error {

	batch := l.batchsize
	if l.cnn != nil {
		return l.cnn.UpdateWeights(l.h.Handler, batch, epoch)
	}

	if l.cnntranspose != nil {
		return l.cnntranspose.UpdateWeights(l.h.Handler, batch, epoch)

	}
	if l.batch != nil {
		return l.batch.UpdateWeights(l.h.Handler, batch, epoch)
	}
	if l.activation != nil {
		if l.activation.TrainersNeeded() > 0 {
			return l.activation.UpdateWeights(l.h.Handler, batch, epoch)

		}

	}
	if l.other != nil {
		return l.other.UpdateWeights(l.h.Handler)
	}
	return nil

}

func (l *Layer) l1l2loss() (l1, l2 float32) {

	if l.cnn != nil {
		return l.cnn.L1L2Loss()
	}

	if l.cnntranspose != nil {
		return l.cnntranspose.L1L2Loss()

	}
	return -123, -123
}
