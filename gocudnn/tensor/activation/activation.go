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

//OpInfo contains the necissary information to build an activation Ops
type OpInfo struct {
	Mode    gocudnn.ActivationMode `json:"Mode"`
	NanProp gocudnn.PropagationNAN `json:"NanProp"`
	Coef    float64                `json:"Coef"`
}

//Flags returns the flags that are needed to create an Activation struct
func Flags() (gocudnn.ActivationModeFlag, gocudnn.PropagationNANFlag) {
	return gocudnn.ActivationModeFlag{}, gocudnn.PropagationNANFlag{}
}

//Stage builds and returns *Op from the info inside of the info type
func (input OpInfo) Stage() (*Ops, error) {
	return StageOperation(input.Mode, input.NanProp, input.Coef)
}

//StageOperation creates an activation struct given the properties passed in function
func StageOperation(mode gocudnn.ActivationMode, nan gocudnn.PropagationNAN, coef float64) (*Ops, error) {
	var hlp gocudnn.Activation
	x, err := hlp.NewActivationDescriptor(mode, nan, coef)
	return &Ops{
		desc: x,
	}, err
}

//Info returns the Info struct.  (Made for saving to a json file at a higher level)
func (act *Ops) Info() (OpInfo, error) {
	var x OpInfo
	var err error
	x.Mode, x.NanProp, x.Coef, err = act.Properties()
	return x, err
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
	a := gocudnn.CScalarByDataType(dtypex, alpha)
	b := gocudnn.CScalarByDataType(dtypex, beta)
	if a == nil || b == nil {
		return errors.New("Unsupported Datatype for either alpha or beta")
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
	a := gocudnn.CScalarByDataType(dtypedx, alpha)
	b := gocudnn.CScalarByDataType(dtypedx, beta)
	if a == nil || b == nil {
		return errors.New("Unsupported Datatype for either alpha or beta")
	}

	return act.helper.Funcs.ActivationBackward(handle, act.desc, a, y.TD(), y.Memer(), dy.TD(), dy.Memer(), x.TD(), x.Memer(), b, dx.TD(), dx.Memer())
}

//Destroy destroys the cuda allocated memory associated with Activation
func (act *Ops) Destroy() error {

	return act.desc.DestroyDescriptor()
}
