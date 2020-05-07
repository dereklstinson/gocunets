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
	id           int64
	name         string
	h            *Handle
	activation   *activation.Layer
	cnn          *cnn.Layer
	pool         *pooling.Layer
	drop         *dropout.Layer
	batch        *batchnorm.Layer
	reshape      *reshape.Layer
	cnntranspose *cnntranspose.Layer
	workspacefwd *nvidia.Malloced
	workspacebwd *nvidia.Malloced
	workspacebwf *nvidia.Malloced
	batchsize    int
	other        Operation
	x, dx, y, dy *Tensor
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

//GetTensorX Gets x tensor
func (l *Layer) GetTensorX() *Tensor {
	return l.x
}

//GetTensorDX Gets dx tensor
func (l *Layer) GetTensorDX() *Tensor {
	return l.dx
}

//GetTensorY Gets y tensor
func (l *Layer) GetTensorY() *Tensor {
	return l.y
}

//GetTensorDY Gets dy tensor
func (l *Layer) GetTensorDY() *Tensor {
	return l.dy
	//return m.dy
}

//SetTensorX sets x tensor
func (l *Layer) SetTensorX(x *Tensor) {
	l.x = x
}

//SetTensorDX sets dx tensor
func (l *Layer) SetTensorDX(dx *Tensor) {
	l.dx = dx
}

//SetTensorY sets y tensor
func (l *Layer) SetTensorY(y *Tensor) {
	l.y = y

}

//SetTensorDY sets dy tensor
func (l *Layer) SetTensorDY(dy *Tensor) {
	l.dy = dy

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

//LoadTrainer loads the trainer to the layer
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

	return errors.New("in bedded error doesn't support trainers")
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
