package gocunets

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
)

func (l *layer) backpropfilter(handle *cudnn.Handler, wspacefilter *nvidia.Malloced, x, y *layers.IO) error {
	var err error
	if l.cnn != nil {
		err = l.cnn.BackPropFilter(handle, wspacefilter, x, y)
		if err != nil {
			println("bpfd error in cnn")
			return err
		}

		return nil

	}

	if l.activation != nil {
		err = l.activation.BackProp(handle, x, y)
		if err != nil {
			println("bpfd error in activation")
			return err
		}

		return nil
	}

	if l.softmax != nil {
		err = l.softmax.BackProp(handle, x, y)
		if err != nil {
			println("bpfd error in softmax")
			return err
		}

		return nil
	}

	if l.drop != nil {
		err = l.drop.BackProp(handle, x, y)
		if err != nil {
			println("bpfd error in drop")
			return err
		}

		return nil
	}

	if l.pool != nil {
		err = l.pool.BackProp(handle, x, y)
		if err != nil {
			println("bpfd error in pool")
			return err
		}

		return nil
	}

	if l.reshape != nil {
		err = l.reshape.BackProp(handle, x, y)
		if err != nil {
			println("bpfd error in reshape")
			return err
		}

		return nil
	}
	if l.batch != nil {
		err = l.batch.BackProp(handle, x, y)
		if err != nil {
			println("bpfd error in batch")
			return err
		}

		return nil
	}
	if l.cnntranspose != nil {
		err = l.cnntranspose.BackPropFilter(handle, wspacefilter, x, y)
		if err != nil {
			println("bpfd error in cnntranspose")
			return err
		}

		return nil
	}
	return errors.New("Layer Not Set Up")

}

//BackProp does the backprop of a layer
func (l *layer) backpropfilterdata(handle *cudnn.Handler, wspacefwd, wspacedata, wspacefilter *nvidia.Malloced, x, y *layers.IO) error {
	err := handle.Sync()
	if err != nil {
		return err
	}
	if l.cnn != nil {
		err = l.cnn.BackPropFilterData(handle, wspacedata, wspacefilter, x, y)
		if err != nil {
			println("bpfd error in cnn")
			return err
		}
		err = handle.Sync()
		if err != nil {
			println("bpfd error in cnn sync")
		}
		return nil

	}

	if l.activation != nil {
		err = l.activation.BackProp(handle, x, y)
		if err != nil {
			println("bpfd error in activation")
			return err
		}
		err = handle.Sync()
		if err != nil {
			println("bpfd error in activation sync")
		}
		return nil
	}

	if l.softmax != nil {
		err = l.softmax.BackProp(handle, x, y)
		if err != nil {
			println("bpfd error in softmax")
			return err
		}
		err = handle.Sync()
		if err != nil {
			println("bpfd error in softmax sync")
		}
		return nil
	}

	if l.drop != nil {
		err = l.drop.BackProp(handle, x, y)
		if err != nil {
			println("bpfd error in drop")
			return err
		}
		err = handle.Sync()
		if err != nil {
			println("bpfd error in drop sync")
		}
		return nil
	}

	if l.pool != nil {
		err = l.pool.BackProp(handle, x, y)
		if err != nil {
			println("bpfd error in pool")
			return err
		}
		err = handle.Sync()
		if err != nil {
			println("bpfd error in pool sync")
		}
		return nil
	}

	if l.reshape != nil {
		err = l.reshape.BackProp(handle, x, y)
		if err != nil {
			println("bpfd error in reshape")
			return err
		}
		err = handle.Sync()
		if err != nil {
			println("bpfd error in reshape sync")
		}
		return nil
	}
	if l.batch != nil {
		err = l.batch.BackProp(handle, x, y)
		if err != nil {
			println("bpfd error in batch")
			return err
		}
		err = handle.Sync()
		if err != nil {
			println("bpfd error in batch sync")
		}
		return nil
	}
	if l.cnntranspose != nil {
		err = l.cnntranspose.BackPropFilterData(handle, wspacefwd, wspacefilter, x, y)
		if err != nil {
			println("bpfd error in cnntranspose")
			return err
		}
		err = handle.Sync()
		if err != nil {
			println("bpfd error in cnntranspose sync")
		}
		return nil
	}
	return errors.New("Layer Not Set Up")
}
