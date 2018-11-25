package gocunets

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

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
