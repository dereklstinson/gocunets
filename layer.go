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
