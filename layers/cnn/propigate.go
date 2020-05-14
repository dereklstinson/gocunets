package cnn

import (
	"fmt"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/layers"
	"github.com/dereklstinson/gocunets/utils"
)

//func (c *Layer)GetStats(handle *cudnn.Handler)

//ForwardProp performs the ForwardProp
func (c *Layer) ForwardProp(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.Tensor) error {
	err := c.conv.Forward(handle, c.fwd.alpha,
		x.Volume,
		c.w.Volume,
		wspace,
		c.fwd.beta,
		y.Volume,
	)
	if err != nil {
		return err
	}
	err = y.Volume.AddTo(handle, c.bias.Volume, 1.0, 1.0)
	if err != nil {
		fmt.Println("y: ", y)
		fmt.Println("bias:", c.bias)
		return err
	}
	return nil
}

//BackPropFilterData does the backprop for the data and the filter
//
//Dx and X can be the same.
func (c *Layer) BackPropFilterData(handle *cudnn.Handler, wspacedata, wspacefilter *nvidia.Malloced, x, dx, dy *layers.Tensor) error {
	var err error
	if x == nil {
		panic("X is nil")
	}
	if dy == nil {
		panic("dy is nil")
	}
	if dx != nil {
		err = handle.Stream().Sync()
		if err != nil {
			return err
		}
		err = c.BackPropData(handle, wspacedata, dx, dy)
		if err != nil {
			return err
		}
	}

	err = handle.Stream().Sync()
	if err != nil {
		return err
	}
	err = c.BackPropFilter(handle, wspacefilter, x, dy)
	if err != nil {
		return err
	}
	err = handle.Stream().Sync()
	if err != nil {
		return err
	}

	return nil
}

//BackPropData performs the BackPropData
func (c *Layer) BackPropData(handle *cudnn.Handler, wspace *nvidia.Malloced, dx, dy *layers.Tensor) error {
	if dx == nil {
		return nil
	}
	return c.conv.BackwardData(
		handle,
		c.bwdd.alpha,
		c.w.Volume,
		dy.Volume,
		wspace,
		c.bwdd.beta,
		dx.Volume,
	)

}

//BackPropFilter does the backward propagation for the filter You will pass a handle workspace memory x,dy layer.io
func (c *Layer) BackPropFilter(handle *cudnn.Handler, wspace *nvidia.Malloced, x, dy *layers.Tensor) error {
	err := c.conv.BackwardFilter(
		handle,
		c.bwdf.alpha,
		x.Volume,
		dy.Volume,
		wspace,
		c.bwdf.beta,
		c.dw.Volume)
	if err != nil {
		return utils.ErrorWrapper("Filter", err)
	}

	err = c.conv.BackwardBias(
		handle,
		c.bwdf.alpha,
		dy.Volume,
		c.bwdf.beta,
		c.dbias.Volume)
	if err != nil {
		return utils.ErrorWrapper("Bias", err)
	}
	return nil
}
