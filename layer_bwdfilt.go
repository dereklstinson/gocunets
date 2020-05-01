package gocunets

import (
	"errors"

	"github.com/dereklstinson/gocunets/layers"
	//"github.com/dereklstinson/gocunets/layers"
)

func (l *Layer) backpropfilter() error {
	//	return l.h.w.Work(func() error {
	var err error
	x, dx, y, dy := l.x.Tensor, l.dx.Tensor, l.y.Tensor, l.dy.Tensor

	if l.cnn != nil {
		err = l.cnn.BackPropFilter(l.h.Handler, l.workspacebwf, x, y)
		if err != nil {
			println("bpfd error in cnn")
			return err
		}

		return nil

	}
	if l.activation != nil {
		if l.activation.ContainsWeights() {
			err = l.activation.BackProp(l.h.Handler, x, dx, y, dy)
			if err != nil {
				println("bpfd error in activation")
				return err
			}

		}
		return nil
	}
	if l.drop != nil {
		/*
			err = l.drop.BackProp(handle, x, y)
			if err != nil {
				println("bpfd error in drop")
				return err
			}
		*/
		return nil
	}

	if l.pool != nil {
		/*
			err = l.pool.BackProp(handle, x, y)
			if err != nil {
				println("bpfd error in pool")
				return err
			}
		*/
		return nil
	}

	if l.reshape != nil {
		/*
			err = l.reshape.BackProp(handle, x, y)
			if err != nil {
				println("bpfd error in reshape")
				return err
			}
		*/
		return nil
	}
	if l.batch != nil {
		/*
			err = l.batch.BackProp(handle, x, y)
			if err != nil {
				println("bpfd error in batch")
				return err
			}
		*/
		return nil
	}
	if l.cnntranspose != nil {
		err = l.cnntranspose.BackPropFilter(l.h.Handler, l.workspacebwd, x, y)
		if err != nil {
			println("bpfd error in cnntranspose")
			return err
		}

		return nil
	}
	return errors.New("Layer Not Set Up")

	//})

}

//BackProp does the backprop of a layer
func (l *Layer) backpropfilterdata() error {
	//return l.h.w.Work(func() error {
	err := l.h.Sync()
	if err != nil {
		return err
	}
	var x, dx, y, dy *layers.Tensor
	if l.dx != nil {
		dx = l.dx.Tensor
	} else {
		dx = nil
	}
	if l.x != nil {
		x = l.x.Tensor
	}
	if l.dy != nil {
		dy = l.dy.Tensor
	}
	if l.y != nil {
		y = l.y.Tensor
	}

	wspacedata, wspacefilter := l.workspacebwd, l.workspacebwf
	if l.cnn != nil {
		err = l.cnn.BackPropFilterData(l.h.Handler, wspacedata, wspacefilter, x, dx, dy)
		if err != nil {
			println("bpfd error in cnn")
			return err
		}
		err = l.h.Sync()
		if err != nil {
			println("bpfd error in cnn sync")
		}
		return nil

	}

	if l.activation != nil {
		err = l.activation.BackProp(l.h.Handler, x, dx, y, dy)
		if err != nil {
			println("bpfd error in activation")
			return err
		}
		err = l.h.Sync()
		if err != nil {
			println("bpfd error in activation sync")
		}
		return nil
	}

	if l.drop != nil {
		err = l.drop.BackProp(l.h.Handler, x, y)
		if err != nil {
			println("bpfd error in drop")
			return err
		}
		err = l.h.Sync()
		if err != nil {
			println("bpfd error in drop sync")
		}
		return nil
	}

	if l.pool != nil {
		err = l.pool.BackProp(l.h.Handler, x, dx, y, dy)
		if err != nil {
			println("bpfd error in pool")
			return err
		}
		err = l.h.Sync()
		if err != nil {
			println("bpfd error in pool sync")
		}
		return nil
	}

	if l.reshape != nil {
		err = l.reshape.BackProp(l.h.Handler, x, y)
		if err != nil {
			println("bpfd error in reshape")
			return err
		}
		err = l.h.Sync()
		if err != nil {
			println("bpfd error in reshape sync")
		}
		return nil
	}
	if l.batch != nil {
		err = l.batch.BackProp(l.h.Handler, x, dx, dy)
		if err != nil {
			println("bpfd error in batch")
			return err
		}
		err = l.h.Sync()
		if err != nil {
			println("bpfd error in batch sync")
		}
		return nil
	}
	if l.cnntranspose != nil {
		err = l.cnntranspose.BackPropFilterData(l.h.Handler, wspacedata, wspacefilter, x, dx, dy)
		if err != nil {
			return err
		}
		err = l.h.Sync()
		if err != nil {
			println("bpfd error in cnntranspose sync")
		}
		return nil
	}
	return errors.New("Layer Not Set Up")

	//	})

}
