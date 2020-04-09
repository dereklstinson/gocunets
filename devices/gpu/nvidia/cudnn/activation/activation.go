package activation

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/xtra"
)

//Ops is the non linear function that is used in neural networks. This structure holds the information used to performing the activation function.
type Ops struct {
	desc  *gocudnn.ActivationD
	xdesc *xtra.XActivationD
	mode  Mode
	nan   gocudnn.NANProp
}

//Stage creates an activation struct given the properties passed in function
func Stage(handle *cudnn.Handler, mode Mode, dtype gocudnn.DataType, nan gocudnn.NANProp, coef float64) (*Ops, error) {

	var mflg Mode
	x, err := gocudnn.CreateActivationDescriptor()
	if err != nil {
		return nil, err
	}
	switch mode {
	case mflg.Threshhold():
		desc, err := xtra.NewXActivationDescriptor(handle.XHandle(), mode.x(), dtype, nan, coef)
		if err != nil {
			return nil, err
		}
		return &Ops{
			xdesc: desc,
			mode:  mode,
		}, err
	case mflg.Leaky():
		desc, err := xtra.NewXActivationDescriptor(handle.XHandle(), mode.x(), dtype, nan, coef)
		if err != nil {
			return nil, err
		}
		return &Ops{
			xdesc: desc,
			mode:  mode,
		}, err
	case mflg.PRelu():
		desc, err := xtra.NewXActivationDescriptor(handle.XHandle(), mode.x(), dtype, nan, coef)
		if err != nil {
			return nil, err
		}
		return &Ops{
			xdesc: desc,
			mode:  mode,
		}, err
	case mflg.ClippedRelu():
		x.Set(mode.c(), nan, coef)
		return &Ops{
			desc: x,
			mode: mode,
		}, err
	case mflg.Elu():
		x.Set(mode.c(), nan, coef)
		return &Ops{
			desc: x,
			mode: mode,
		}, err

	case mflg.Relu():
		x.Set(mode.c(), nan, coef)
		return &Ops{
			desc: x,
			mode: mode,
		}, err
	case mflg.Sigmoid():
		x.Set(mode.c(), nan, coef)
		return &Ops{
			desc: x,
			mode: mode,
		}, err
	case mflg.Tanh():
		x.Set(mode.c(), nan, coef)
		return &Ops{
			desc: x,
			mode: mode,
		}, err

	}
	return nil, errors.New("Not supported activation")
}

//Info returns the Info struct.  (Made for saving to a json file at a higher level)
func (act *Ops) Info() (OpInfo, error) {

	amode, propnan, coef, err := act.Properties()

	return OpInfo{
		Mode:    Mode(amode),
		NanProp: (propnan),
		Coef:    coef,
	}, err
}

//Mode returns the activation Mode
func (act *Ops) Mode() Mode {
	return act.mode
}

//Properties returns the values that were used to Create the Activation struct
func (act *Ops) Properties() (Mode, gocudnn.NANProp, float64, error) {
	a, b, c, err := act.desc.Get()
	return Mode{
		m: a,
	}, b, c, err

}

//FwdProp is the forward propigation function for the Activation struct
func (act *Ops) FwdProp(
	handle *cudnn.Handler,
	alpha float64,
	x *tensor.Volume,
	beta float64,
	y *tensor.Volume,
	negcoef *tensor.Volume,
	thresh *tensor.Volume,
	poscoef *tensor.Volume) error {
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
	a := alpha
	b := beta
	var mflg Mode
	switch act.mode {
	case mflg.Threshhold():
		return act.xdesc.ForwardProp(handle.XHandle(), x.TD(), x, y.TD(), y, negcoef, thresh, poscoef, a, b)
	case mflg.Leaky():
		return act.xdesc.ForwardProp(handle.XHandle(), x.TD(), x, y.TD(), y, nil, nil, nil, a, b)
	case mflg.PRelu():
		return act.xdesc.ForwardProp(handle.XHandle(), x.TD(), x, y.TD(), y, negcoef, nil, nil, a, b)
	case mflg.ClippedRelu():
		return act.desc.Forward(handle.Cudnn(), a, x.TD(), x, b, y.TD(), y)
	case mflg.Elu():
		return act.desc.Forward(handle.Cudnn(), a, x.TD(), x, b, y.TD(), y)
	case mflg.Relu():
		return act.desc.Forward(handle.Cudnn(), a, x.TD(), x, b, y.TD(), y)
	case mflg.Sigmoid():
		return act.desc.Forward(handle.Cudnn(), a, x.TD(), x, b, y.TD(), y)
	case mflg.Tanh():
		return act.desc.Forward(handle.Cudnn(), a, x.TD(), x, b, y.TD(), y)

	}
	return errors.New("unsupported Activation Mode")
}

//BwdProp is the backwards propigation of the activation struct
//
//In-place operation is allowed for this routine; meaning dy and dx pointers may be equal. However, this requires the corresponding tensor descriptors to be identical (particularly, the strides of the input and output must match for an in-place operation to be allowed).
//
//All tensor formats are supported for 4 and 5 dimensions, however, the best performance is obtained when the strides of yDesc and xDesc are equal and HW-packed. For more than 5 dimensions the tensors must have their spatial dimensions packed.
func (act *Ops) BwdProp(
	handle *cudnn.Handler,
	alpha float64,
	y *tensor.Volume,
	dy *tensor.Volume,
	x *tensor.Volume,
	beta float64,
	dx *tensor.Volume,
	negcoef *tensor.Volume,
	dnegcoef *tensor.Volume,
	thresh *tensor.Volume,
	dthresh *tensor.Volume,
	poscoef *tensor.Volume,
	dposcoef *tensor.Volume) error {
	_, dtypedx, _, err := dx.Properties()
	if err != nil {
		return err
	}
	_, dtypex, _, err := x.Properties()
	if err != nil {
		return err
	}
	_, dtypey, _, err := y.Properties()
	if err != nil {
		return err
	}
	_, dtypedy, _, err := dy.Properties()
	if err != nil {
		return err
	}
	if dtypedx != dtypey || dtypedx != dtypedy || dtypedx != dtypex {
		return errors.New("output type not matching input type")
	}
	a := alpha
	b := beta
	var mflg Mode
	switch act.mode {
	case mflg.Threshhold():
		return act.xdesc.BackProp(handle.XHandle(), x.TD(), x, dx.TD(), dx, dy.TD(), dy, negcoef, dnegcoef, thresh, dthresh, poscoef, dposcoef, a, b)
	case mflg.Leaky():
		return act.xdesc.BackProp(handle.XHandle(), x.TD(), x, dx.TD(), dx, dy.TD(), dy, nil, nil, nil, nil, nil, nil, a, b)
	case mflg.PRelu():
		return act.xdesc.BackProp(handle.XHandle(), x.TD(), x, dx.TD(), dx, dy.TD(), dy, negcoef, dnegcoef, nil, nil, nil, nil, a, b)
	case mflg.ClippedRelu():
		return act.desc.Backward(handle.Cudnn(), a, y.TD(), y, dy.TD(), dy, x.TD(), x, b, dx.TD(), dx)
	case mflg.Elu():
		return act.desc.Backward(handle.Cudnn(), a, y.TD(), y, dy.TD(), dy, x.TD(), x, b, dx.TD(), dx)
	case mflg.Relu():
		return act.desc.Backward(handle.Cudnn(), a, y.TD(), y, dy.TD(), dy, x.TD(), x, b, dx.TD(), dx)
	case mflg.Sigmoid():
		return act.desc.Backward(handle.Cudnn(), a, y.TD(), y, dy.TD(), dy, x.TD(), x, b, dx.TD(), dx)
	case mflg.Tanh():
		return act.desc.Backward(handle.Cudnn(), a, y.TD(), y, dy.TD(), dy, x.TD(), x, b, dx.TD(), dx)

	}
	return errors.New("unsupported Activation Mode")

}

//Destroy destroys the cuda allocated memory associated with Activation
func (act *Ops) Destroy() error {

	return act.desc.Destroy()
}
