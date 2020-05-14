package gocunets

import (
	"errors"

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
)

//Operation is a generic operation that a layer uses.
//
//The forward and backward don't need to use all the x,dx,y,and dy, but they do need to be passed.
//
type Operation interface {
	Forward(handle *cudnn.Handler, x, dx, y, dy *layers.Tensor) error
	Inference(handle *cudnn.Handler, x, y *layers.Tensor) error
	Backward(handle *cudnn.Handler, x, dx, y, dy *Tensor) error
	GetWeights() []*layers.Tensor
	GetDeltaWeights() []*layers.Tensor
	SetOtherScalars(alpha, beta float64)
	SetForwardScalars(alpha, beta float64)
	SetBackwardScalars(alpha, beta float64)
	GetOutputDims(input *layers.Tensor) ([]int32, error)
	InitHiddenValues() (err error)
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

//GetWeights gets the weights of the layer
//will return nil if layer doesn't have weights
func (l *Layer) GetWeights() []*Tensor {
	if l.cnn != nil {
		x := l.cnn.GetWeights()
		tensors := make([]*Tensor, len(x))
		for i := range tensors {
			tensors[i] = new(Tensor)
			tensors[i].Tensor = x[i]
			tensors[i].id = l.id
		}
		return tensors
	}
	if l.cnntranspose != nil {
		x := l.cnntranspose.GetWeights()
		tensors := make([]*Tensor, len(x))
		for i := range tensors {
			tensors[i] = new(Tensor)
			tensors[i].Tensor = x[i]
			tensors[i].id = l.id
		}
		return tensors
	}
	if l.activation != nil {
		x := l.activation.GetWeights()
		if len(x) == 0 {
			return nil
		}
		tensors := make([]*Tensor, len(x))
		for i := range tensors {
			tensors[i] = new(Tensor)
			tensors[i].Tensor = x[i]
			tensors[i].id = l.id
		}
		return tensors
	}
	if l.other != nil {

		x := l.other.GetWeights()
		if len(x) == 0 {
			return nil
		}
		tensors := make([]*Tensor, len(x))
		for i := range tensors {
			tensors[i] = new(Tensor)
			tensors[i].Tensor = x[i]
			tensors[i].id = l.id
		}
		return tensors
	}
	return nil

}

//GetDeltaWeights gets the weights of the layer
//will return nil if layer doesn't have weights
func (l *Layer) GetDeltaWeights() []*Tensor {
	if l.cnn != nil {
		x := l.cnn.GetDeltaWeights()
		tensors := make([]*Tensor, len(x))
		for i := range tensors {
			tensors[i] = new(Tensor)
			tensors[i].Tensor = x[i]
			tensors[i].id = l.id
		}

		return tensors
	}
	if l.cnntranspose != nil {
		x := l.cnntranspose.GetDeltaWeights()
		tensors := make([]*Tensor, len(x))
		for i := range tensors {
			tensors[i] = new(Tensor)
			tensors[i].Tensor = x[i]
			tensors[i].id = l.id
		}
		return tensors
	}
	if l.activation != nil {
		x := l.activation.GetDeltaWeights()
		if len(x) == 0 {
			return nil
		}
		tensors := make([]*Tensor, len(x))
		for i := range tensors {
			tensors[i] = new(Tensor)
			tensors[i].Tensor = x[i]
			tensors[i].id = l.id
		}
		return tensors
	}
	if l.other != nil {

		x := l.other.GetDeltaWeights()
		if len(x) == 0 {
			return nil
		}
		tensors := make([]*Tensor, len(x))
		for i := range tensors {
			tensors[i] = new(Tensor)
			tensors[i].Tensor = x[i]
			tensors[i].id = l.id
		}
		return tensors
	}
	return nil

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

//Module is a wrapper around a neural network or set of operations
type tempModule interface {
	ID() int64
	Forward() error
	Backward() error
	Inference() error
	InitHiddenLayers() (err error)
	InitWorkspace() (err error)
	GetTensorX() (x *Tensor)
	GetTensorDX() (dx *Tensor)
	GetTensorY() (y *Tensor)
	GetTensorDY() (dy *Tensor)
	SetTensorX(x *Tensor)
	SetTensorDX(dx *Tensor)
	SetTensorY(y *Tensor)
	SetTensorDY(dy *Tensor)
}

//InitHiddenLayers inits hidden values after they are all set.  It doesn't init the output tensor though.
func (l *Layer) InitHiddenLayers() (err error) {
	if l.x == nil {
		return errors.New("Input X Tensor not set")
	}
	if l.cnn != nil {
		err := l.cnn.MakeRandom(l.h.Handler, l.x.Dims())
		if err != nil {
			return err
		}
	} else if l.cnntranspose != nil {
		err := l.cnntranspose.MakeRandom(l.h.Handler, l.x.Dims())
		if err != nil {
			return err
		}
	} else if l.other != nil {
		return l.other.InitHiddenValues()

	}
	return nil
}

func (l *Layer) Inference() error {
	return l.inference(l.h.Handler, l.workspacefwd, l.workspacebwd)
}

/*
func (l *Layer) InitWorkspace() (err error) {
	//if l.cnn.GetFwdAlgoPerfList()

}
*/
//FindOutputDims gets the dims of the output tensor
func (l *Layer) OutputDims() (output []int32, err error) {
	if l.x == nil {
		return nil, errors.New("(l *Layer) FindOutputDims():x tensor not set")
	}
	if l.cnn != nil {
		return l.cnn.FindOutputDims(l.x.Tensor)
	}
	if l.pool != nil {
		return l.pool.GetOutputDims(l.x.Tensor)
	}
	if l.batch != nil {
		output = make([]int32, len(l.x.Dims()))
		copy(output, l.x.Dims())
		return output, nil
	}
	if l.cnntranspose != nil {
		return l.cnntranspose.FindOutputDims(l.x.Tensor)
	}
	if l.activation != nil {
		output = make([]int32, len(l.x.Dims()))
		copy(output, l.x.Dims())
		return output, nil
	}
	if l.drop != nil {
		output = make([]int32, len(l.x.Dims()))
		copy(output, l.x.Dims())
		return output, nil
	}
	if l.other != nil {
		return l.other.GetOutputDims(l.x.Tensor)
	}
	return nil, errors.New("Unsupported Layer")
}

func wraplayer(input interface{}) (hidden *Layer, ios int) {
	switch l := input.(type) {

	case *activation.Layer:

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
