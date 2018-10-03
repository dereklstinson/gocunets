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
	"github.com/dereklstinson/GoCuNets/layers/xactivation"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//layer contains a layer // ie activation or cnn or fcnn
type layer struct {
	xactivation *xactivation.Layer
	activation  *activation.Layer
	cnn         *cnn.Layer
	fcnn        *fcnn.Layer
	softmax     *softmax.Layer
	pool        *pooling.Layer
	drop        *dropout.Layer
	batchnorm   *batchnorm.Layer
}
type IO struct {
	parents  []int
	children []int
	mem      *layers.IO
}
type Network struct {
}
type Module struct {
	index int64
	layer []*layer
	mem   []*layers.IO
}
type Handles struct {
	cudnn   *gocudnn.Handle
	xhandle *gocudnn.XHandle
}

func CreateHandles(dev *gocudnn.Device, trainingfolder string) (*Handles, error) {

	x := gocudnn.NewHandle()

	y, err := gocudnn.Xtra{}.MakeXHandle(trainingfolder, dev)
	if err != nil {
		return nil, err
	}
	return &Handles{
		cudnn:   x,
		xhandle: y,
	}, nil
}

//ForwardProp does the forward prop for a prebuilt module
func (m *Module) ForwardProp(handle Handles, stream *gocudnn.Stream, wspace *gocudnn.Malloced, x, y *layers.IO) error {

	err := m.layer[0].forwardprop(handle, wspace, x, m.mem[0])
	if err != nil {
		return err
	}
	lnum := len(m.layer)
	for i := 1; i < lnum-1; i++ {
		err = m.layer[i].forwardprop(handle, wspace, m.mem[i-1], m.mem[i])
		if err != nil {
			return err
		}
	}
	//for m.layer[lnum-1] is the last layer in the slice and m.mem[lnum-2] should be the output from the layer before.
	err = m.layer[lnum-1].forwardprop(handle, wspace, m.mem[lnum-2], y)
	if err != nil {
		return err
	}

	return stream.Sync()
}

//ForwardProp does the forward prop for a layer
func (l *layer) forwardprop(handle Handles, wpace gocudnn.Memer, x, y *layers.IO) error {
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
	if l.xactivation != nil {
		return l.xactivation.ForwardProp(handle.xhandle, x, y)
	}
	return errors.New("Layer Not Set Up")
}

//BackProp does the backprop of a module
func (m *Module) BackProp(handle Handles, stream *gocudnn.Stream, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	lnum := len(m.layer)
	err := m.layer[lnum-1].backprop(handle, wspace, m.mem[lnum-2], y)

	if err != nil {
		return err
	}

	for i := lnum - 2; i > 1; i-- {
		err = m.layer[i].backprop(handle, wspace, m.mem[i-1], m.mem[i])
		if err != nil {
			return err
		}
	}

	err = m.layer[0].backprop(handle, wspace, x, m.mem[0])
	if err != nil {
		return err
	}

	return stream.Sync()
}

//BackProp does the backprop of a layer
func (l *layer) backprop(handle Handles, wpace gocudnn.Memer, x, y *layers.IO) error {
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
	if l.xactivation != nil {
		return l.xactivation.BackProp(handle.xhandle, x, y)
	}
	return errors.New("Layer Not Set Up")
}

//UpdateWeights updates the weights of a module
func (m *Module) UpdateWeights(handle *Handles) error {
	for i := 0; i < len(m.layer); i++ {
		err := m.layer[i].updateWeights(handle)
		if err != nil {
			return err
		}
	}
	return nil
}

//UpdateWeights updates the weights of layer
func (l *layer) updateWeights(handle *Handles) error {
	if l.cnn != nil {
		return l.cnn.UpdateWeights(handle.xhandle)
	}
	if l.fcnn != nil {
		return l.fcnn.UpdateWeights(handle.xhandle)
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
