package gocunets

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

func (l *layer) inference(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {

	err := handle.Sync()
	if err != nil {
		return err
	}
	if l.cnn != nil {
		err = l.cnn.ForwardProp(handle, wspace, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}

	if l.drop != nil {
		err = l.drop.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.activation != nil {

		err = l.activation.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.softmax != nil {
		err = l.softmax.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.pool != nil {
		err = l.pool.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}

	if l.reshape != nil {
		err = l.reshape.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()

	}
	if l.batch != nil {
		err = l.batch.ForwardInference(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
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

//ForwardProp does the forward prop for a layer
func (l *layer) forwardprop(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {

	err := handle.Sync()
	if err != nil {
		return err
	}
	if l.cnn != nil {
		err = l.cnn.ForwardProp(handle, wspace, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}

	if l.drop != nil {
		err = l.drop.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.activation != nil {

		err = l.activation.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.softmax != nil {
		err = l.softmax.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.pool != nil {
		err = l.pool.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
	}

	if l.reshape != nil {
		err = l.reshape.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()

	}
	if l.batch != nil {
		err = l.batch.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		return handle.Sync()
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
