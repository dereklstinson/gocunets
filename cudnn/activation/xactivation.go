package activation

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Stage creates an activation struct given the properties passed in function
func xstage(h *gocudnn.XHandle, amode gocudnn.XActivationMode, tmode gocudnn.TrainingMode, dtype gocudnn.DataType, invcoef float64) (*Ops, error) {
	var xtra gocudnn.Xtra

	desc, err := xtra.NewXActivationDescriptor(h, amode, tmode, dtype, invcoef)
	if err != nil {
		return nil, err
	}
	return &Ops{
		xdesc: desc,
	}, err
}

//FwdProp is the forward propigation function for the Activation struct
//Alphas will only be in use if the activation descriptor is using it. otherwise it can be nil
func (act *Ops) xfwdprp(
	handle *gocudnn.XHandle,
	x *tensor.Volume,
	y *tensor.Volume,
	alphas *tensor.Volume,
	betas *tensor.Volume) error {
	_, dtypex, _, err := x.Properties()
	if err != nil {
		return err
	}
	_, dtypey, _, err := y.Properties()

	if err != nil {
		return err
	}
	if dtypex != dtypey {
		return errors.New("output type not matching input type")
	}

	if alphas == nil && betas == nil {
		return act.xdesc.ForwardProp(handle, x.TD(), x.Memer(), y.TD(), y.Memer(), nil, nil)
	}
	return act.xdesc.ForwardProp(handle, x.TD(), x.Memer(), y.TD(), y.Memer(), alphas.Memer(), betas.Memer())

}

func (act *Ops) xbwdprop(
	handle *gocudnn.XHandle,
	x *tensor.Volume,
	dx *tensor.Volume,
	dy *tensor.Volume,
	alphas *tensor.Volume,
	dalphas *tensor.Volume,
) error {

	if alphas == nil || dalphas == nil {
		return act.xdesc.BackProp(handle, x.TD(), x.Memer(), dx.TD(), dx.Memer(), dy.TD(), dy.Memer(), nil, nil)
	}
	return act.xdesc.BackProp(handle,
		x.TD(),
		x.Memer(),
		dx.TD(),
		dx.Memer(),
		dy.TD(),
		dy.Memer(),
		alphas.Memer(),
		dalphas.Memer(),
	)

}

//UpdateParams is only used for Xactivation descriptors that support it. (right now only parametric)
func (act *Ops) UpdateParams(
	handle *gocudnn.XHandle,
	batch int,
	alphas *tensor.Volume,
	dalphas *tensor.Volume,
	xsum *tensor.Volume,
	gsum *tensor.Volume,
	l1 *gocudnn.Malloced,
	l2 *gocudnn.Malloced,
	t gocudnn.TrainingParams,
	r gocudnn.RegParams,
) error {
	if gsum == nil || alphas == nil || dalphas == nil {
		return errors.New("Needed Param mem is nil")
	}
	return act.xdesc.UpdateParas(handle, alphas.TD(), alphas.Memer(), dalphas.Memer(), xsum.Memer(), gsum.Memer(), l1, l2, t, r)

}
