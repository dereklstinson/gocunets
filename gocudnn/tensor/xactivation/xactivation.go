package xactivation

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Ops is the non linear function that is used in neural networks. This structure holds the information used to performing the activation function.
type Ops struct {
	desc      *gocudnn.XActivationD
	blocksize uint32
}

//Stage creates an activation struct given the properties passed in function
func Stage(h *gocudnn.XHandle, blocksize uint32, amode gocudnn.XActivationMode, tmode gocudnn.TrainingMode, dtype gocudnn.DataType, coef float64) (*Ops, error) {
	var xtra gocudnn.Xtra

	desc, err := xtra.NewXActivationDescriptor(h, amode, tmode, dtype, coef)
	if err != nil {
		return nil, err
	}
	return &Ops{
		desc:      desc,
		blocksize: blocksize,
	}, err
}

//FwdProp is the forward propigation function for the Activation struct
//Alphas will only be in use if the activation descriptor is using it. otherwise it can be nil
func (act *Ops) FwdProp(
	handle *gocudnn.XHandle,
	x *tensor.Volume,
	y *tensor.Volume,
	alphas *tensor.Volume) error {
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

	var alphaholder gocudnn.Memer
	if alphas == nil {
		alphaholder = nil
	} else {
		alphaholder = alphas.Memer()
	}

	return act.desc.ForwardProp(handle, act.blocksize, x.TD(), x.Memer(), y.TD(), y.Memer(), alphaholder)
}

//BwdProp is the backwards propigation alphas and dalphas can be nil if they are not supported by the descriptor and they both have to be nil or both on.
func (act *Ops) BwdProp(
	handle *gocudnn.XHandle,
	dx *tensor.Volume,
	dy *tensor.Volume,
	alphas *tensor.Volume,
	dalphas *tensor.Volume,
) error {

	if alphas == nil || dalphas == nil {
		return act.desc.BackProp(handle, act.blocksize, dx.TD(), dx.Memer(), dy.TD(), dy.Memer(), nil, nil)
	}
	return act.desc.BackProp(handle, act.blocksize, dx.TD(), dx.Memer(), dy.TD(), dy.Memer(), alphas.Memer(), dalphas.Memer())

}

//UpdateParams is only used for Xactivation descriptors that support it. (right now only parametric)
func (act *Ops) UpdateParams(
	handle *gocudnn.XHandle,
	batch int,
	alphas *tensor.Volume,
	dalphas *tensor.Volume,
	gsum gocudnn.Memer,
	xsum gocudnn.Memer,
	t gocudnn.TrainingParams,
) error {
	if alphas == nil || dalphas == nil || gsum == nil || xsum == nil {
		return errors.New("One or more of the alphas dalphas gsum or xsum is nil")
	}
	return act.desc.UpdateAlphas(handle, act.blocksize, batch, alphas.TD(), alphas.Memer(), dalphas.Memer(), xsum, gsum, t)

}
