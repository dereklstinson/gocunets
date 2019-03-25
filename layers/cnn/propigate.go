package cnn

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/utils"
)

//ForwardProp performs the ForwardProp
func (c *Layer) ForwardProp(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.IO) error {
	err := c.conv.Forward(handle, c.fwd.alpha,
		x.T(),
		c.w.T(),
		wspace,
		c.fwd.beta,
		y.T(),
	)
	if err != nil {
		return err
	}
	return y.T().AddTo(handle, c.bias.T(), 1.0, 1.0)
}

//BackPropFilterData does the backprop for the data and the filter
func (c *Layer) BackPropFilterData(handle *cudnn.Handler, wspacedata, wspacefilter *nvidia.Malloced, x, y *layers.IO) error {
	var err error
	if x.IsInput() == true {
		return c.BackPropFilter(handle, wspacefilter, x, y)
	}
	err = c.BackPropData(handle, wspacedata, x, y)
	if err != nil {
		return err
	}
	err = handle.Stream().Sync()
	if err != nil {
		return err
	}
	return c.BackPropFilter(handle, wspacefilter, x, y)
}

//BackPropData performs the BackPropData
func (c *Layer) BackPropData(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.IO) error {
	if x.IsInput() == true {
		return nil
	}
	return c.conv.BackwardData(
		handle,
		c.bwdd.alpha,
		c.w.T(),
		y.DeltaT(),
		wspace,
		c.bwdd.beta,
		x.DeltaT(),
	)

}

//BackPropFilter does the backward propagation for the filter You will pass a handle workspace memory x,dy layer.io
func (c *Layer) BackPropFilter(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.IO) error {
	err := c.conv.BackwardFilter(
		handle,
		c.bwdf.alpha,
		x.T(),
		y.DeltaT(),
		wspace,
		c.bwdf.beta,
		c.w.DeltaT())
	if err != nil {
		return utils.ErrorWrapper("Filter", err)
	}

	err = c.conv.BackwardBias(
		handle,
		c.bwdf.alpha,
		y.DeltaT(),
		c.bwdf.beta,
		c.bias.DeltaT())
	if err != nil {
		return utils.ErrorWrapper("Bias", err)
	}
	return nil
}
