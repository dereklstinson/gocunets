package gocunets

import (
	"errors"
)

//BackProp does the backprop of a layer
// Transpose workspace backward is actually forward
func (l *Layer) backpropdata() error {
	handle := l.h.Handler
	bwdwspace := l.workspacebwd
	err := handle.Sync()

	if err != nil {
		return err
	}

	if l.cnn != nil {
		dx, dy := l.dx.Tensor, l.dy.Tensor
		err = l.cnn.BackPropData(handle, bwdwspace, dx, dy)
		if err != nil {
			return err
		}
		return handle.Sync()
	}

	if l.activation != nil {
		x := l.x.Tensor
		dx := l.dx.Tensor
		y := l.y.Tensor
		dy := l.dy.Tensor

		err = l.activation.BackProp(handle, x, dx, y, dy)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.drop != nil {
		dx, dy := l.dx.Tensor, l.dy.Tensor
		err = l.drop.BackProp(handle, dx, dy)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.pool != nil {
		x := l.x.Tensor
		dx := l.dx.Tensor
		y := l.y.Tensor
		dy := l.dy.Tensor
		err = l.pool.BackProp(handle, x, dx, y, dy)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.reshape != nil {
		dx, dy := l.dx.Tensor, l.dy.Tensor
		err = l.reshape.BackProp(handle, dx, dy)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.batch != nil {
		x, dx, dy := l.x.Tensor, l.dx.Tensor, l.dy.Tensor
		err = l.batch.BackProp(handle, x, dx, dy)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	if l.cnntranspose != nil {
		dx, dy := l.dx.Tensor, l.dy.Tensor
		err = l.cnntranspose.BackPropData(handle, bwdwspace, dx, dy)
		if err != nil {
			return err
		}
		return handle.Sync()
	}
	return errors.New("Layer Not Set Up")
}
