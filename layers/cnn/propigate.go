package cnn

import (
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//ForwardProp performs the ForwardProp
func (c *Layer) ForwardProp(handle *gocudnn.Handle, wspace gocudnn.Memer, x, y *layers.IO) error {
	err := c.conv.FwdProp(handle, c.fwd.alpha,
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
func (c *Layer) BackPropFilterData(handle *gocudnn.Handle, wspace gocudnn.Memer, x, y *layers.IO) error {
	var err error
	if x.IsInput() == true {
		return c.BackPropFilter(handle, wspace, x, y)
	}
	err = c.BackPropData(handle, wspace, x, y)
	if err != nil {
		return err
	}

	return c.BackPropFilter(handle, wspace, x, y)
}

//BackPropData performs the BackPropData
func (c *Layer) BackPropData(handle *gocudnn.Handle, wspace gocudnn.Memer, x, y *layers.IO) error {
	if x.IsInput() == true {
		return nil
	}
	return c.conv.BwdPropData(
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
func (c *Layer) BackPropFilter(handle *gocudnn.Handle, wspace gocudnn.Memer, x, y *layers.IO) error {
	err := c.conv.BwdPropFilt(
		handle,
		c.bwdf.alpha,
		x.T(),
		y.DeltaT(),
		wspace,
		c.bwdf.beta,
		c.w.DeltaT())
	if err != nil {
		return err
	}

	return c.conv.BwdBias(
		handle,
		c.bwdf.alpha,
		y.DeltaT(),
		c.bwdf.beta,
		c.bias.DeltaT())

}
