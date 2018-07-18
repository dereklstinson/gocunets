package activation

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	"github.com/dereklstinson/GoCudnn"
)

//Ops is the non linear function that is used in neural networks. This structure holds the information used to performing the activation function.
type Ops struct {
	helper gocudnn.Activation
	desc   *gocudnn.ActivationD
}

//Flags returns the flags that are needed to create an Activation struct
func Flags() (gocudnn.ActivationModeFlag, gocudnn.PropagationNANFlag) {
	return gocudnn.ActivationModeFlag{}, gocudnn.PropagationNANFlag{}
}

//Build creates an activation struct given the properties passed in function
func Build(mode gocudnn.ActivationMode, nan gocudnn.PropagationNAN, coef float64) (*Ops, error) {
	var hlp gocudnn.Activation
	x, err := hlp.NewActivationDescriptor(mode, nan, coef)
	return &Ops{
		desc: x,
	}, err
}

//Properties returns the values that were used to Create the Activation struct
func (act *Ops) Properties() (gocudnn.ActivationMode, gocudnn.PropagationNAN, float64, error) {
	return act.desc.GetDescriptor()

}

//FwdProp is the forward propigation function for the Activation struct
func (act *Ops) FwdProp(
	handle *gocudnn.Handle,
	alpha float64,
	x *tensor.Volume,
	beta float64,
	y *tensor.Volume) error {
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
	t := gocudnn.Tensor{}.Flgs.Data
	var a gocudnn.CScalar
	var b gocudnn.CScalar
	switch dtypex {
	case t.Double():
		a = gocudnn.CDouble(alpha)
		b = gocudnn.CDouble(beta)
	case t.Float():
		a = gocudnn.CFloat(alpha)
		b = gocudnn.CFloat(beta)
	case t.Int32():
		a = gocudnn.CInt(alpha)
		b = gocudnn.CInt(beta)
	case t.Int8():
		a = gocudnn.CInt8(alpha)
		b = gocudnn.CInt8(beta)
	case t.UInt8():
		a = gocudnn.CUInt8(alpha)
		b = gocudnn.CUInt8(beta)

	default:
		return errors.New("Not supported Format")
	}
	return act.helper.Funcs.ActivationForward(handle, act.desc, a, x.TD(), x.Memer(), b, y.TD(), y.Memer())
}

//BwdProp is the backwards propigation of the activation struct
func (act *Ops) BwdProp(
	handle *gocudnn.Handle,
	alpha float64,
	y *tensor.Volume,
	dy *tensor.Volume,
	x *tensor.Volume,
	beta float64,
	dx *tensor.Volume,
) error {
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
	t := gocudnn.Tensor{}.Flgs.Data
	var a gocudnn.CScalar
	var b gocudnn.CScalar
	switch dtypedx {
	case t.Double():
		a = gocudnn.CDouble(alpha)
		b = gocudnn.CDouble(beta)
	case t.Float():
		a = gocudnn.CFloat(alpha)
		b = gocudnn.CFloat(beta)
	case t.Int32():
		a = gocudnn.CInt(alpha)
		b = gocudnn.CInt(beta)
	case t.Int8():
		a = gocudnn.CInt8(alpha)
		b = gocudnn.CInt8(beta)
	case t.UInt8():
		a = gocudnn.CUInt8(alpha)
		b = gocudnn.CUInt8(beta)

	default:
		return errors.New("Not supported Format")
	}
	return act.helper.Funcs.ActivationBackward(handle, act.desc, a, y.TD(), y.Memer(), dy.TD(), dy.Memer(), x.TD(), x.Memer(), b, dx.TD(), dx.Memer())
}

//Destroy destroys the cuda allocated memory associated with Activation
func (act *Ops) Destroy() error {

	return act.desc.DestroyDescriptor()
}
