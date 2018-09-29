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
	"github.com/dereklstinson/GoCuNets/layers/softmax"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer contains a layer // ie activation or cnn or fcnn
type Layer struct {
	activation *activation.Layer
	cnn        *cnn.Layer
	fcnn       *fcnn.Layer
	softmax    *softmax.Layer
	pool       *pooling.Layer
	drop       *dropout.Layer
	batchnorm  *batchnorm.Layer
}
type IO struct {
	parents  []int
	children []int
	mem      layers.IO
}
type Module struct {
	index  int64
	layers []Layer
	mem    []layers.IO
}
type Handles struct {
	cudnn   *gocudnn.Handle
	trainer *gocudnn.TrainHandle
}

func CreateHandles(dev *gocudnn.Device, trainingfolder string) (*Handles, error) {

	x := gocudnn.NewHandle()

	y, err := gocudnn.Xtra{}.MakeTrainingHandle(trainingfolder, dev)
	if err != nil {
		return nil, err
	}
	return &Handles{
		cudnn:   x,
		trainer: y,
	}, nil
}

func (l *Layer) ForwardProp(handle Handles, wpace gocudnn.Memer, x, y *layers.IO) error {
	if l.cnn != nil {
		return l.cnn.ForwardProp(handle.cudnn, wpace, x, y)
	}
	if l.fcnn != nil {
		return l.fcnn.ForwardProp(handle.cudnn, x, y)
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
	return errors.New("Layer Not Set Up")
}

func (l *Layer) BackProp(handle Handles, wpace gocudnn.Memer, x, y *layers.IO) error {
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
	if l.pool != nil {
		return l.pool.BackProp(handle.cudnn, x, y)
	}
	return errors.New("Layer Not Set Up")
}

func (l *Layer) UpdateWeights(handle *Handles) error {
	if l.cnn != nil {
		return l.cnn.UpdateWeights(handle.trainer)
	}
	if l.fcnn != nil {
		return l.fcnn.UpdateWeights(handle.trainer)
	}
	return nil
}

/*
func MakeLayer(input interface{}) Layer {
	switch x := input.(type) {

	}
}
func (l *Layer) Setup() error {
	if l.cnn != nil {
		//return l.cnn.(handle.cudnn, wpace, x, y)
	}
	if l.fcnn != nil {
		return l.fcnn.Setup(handle.cudnn, x, y)
	}
	if l.activation != nil {
		return l.Setup.BackProp(handle.cudnn, x, y)
	}
	if l.softmax != nil {
		return l.Setup.BackProp(handle.cudnn, x, y)
	}
	if l.pool != nil {
		return l.pool.Set(handle.cudnn, x, y)
	}
	return errors.New("Layer Not Set Up")
}
*/
