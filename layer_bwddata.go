package gocunets

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
)

//BackProp does the backprop of a layer
// Transpose workspace backward is actually forward
func (l *layer) backpropdata(handle *cudnn.Handler, fwdwspace, bwdwspace *nvidia.Malloced, x, y *layers.IO) error {

	err := handle.Sync()
	if err != nil {
		return err
	}
	if l.cnn != nil {
		err = l.cnn.BackPropData(handle, bwdwspace, x, y)
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
		err = l.cnntranspose.BackPropData(handle, fwdwspace, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	return errors.New("Layer Not Set Up")
}
