package activation

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/activation"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/reduce"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/trainer"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is an activation layer
type Layer struct {
	act                  *activation.Ops
	reduce               *reduce.Ops
	fwd                  Scalars
	bwd                  Scalars
	memmanaged           bool
	updatable            bool
	nanproped            gocudnn.PropagationNAN
	threshandpreludims   []int32
	posmin, posmax       float32
	negmin, negmax       float32
	threshmin, threshmax float32
	negcotrain           trainer.Trainer
	poscotrain           trainer.Trainer
	l1n, l2n, l1p, l2p   float32
	posCoefs             *layers.IO
	negCoefs             *layers.IO
	threshold            *layers.IO
}

//Info is a struct that contains the info that is needed to build the activation layer
type Info struct {
	Ops              activation.OpInfo `json:"Ops"`
	Fwd              Scalars           `json:"Fwd"`
	Bwd              Scalars           `json:"Bwd"`
	OutputMemManaged bool              `json:"OutputMemManaged"`
}

//Scalars are the scalars used in the activation operation
type Scalars struct {
	Alpha float64 `json:"A"`
	Beta  float64 `json:"B"`
}

const defaultalpha = float64(1)
const defaultbeta = float64(0)
const defaultcoef = float64(6)
const defaultnanprop = cudnn.NanMode(0) //NotPropigateNAN
const defaultleakycoef = float64(.01)

//Setup takes default settings for coef (6) and NottPropNan. alpha =1 ,beta =0
//You can change the values by using the Layer methods.
//The way that alpha and beta work is this Y=(alpha *ActivationOp)+(beta*Y).
//It's best to keep the defaults of alpha and beta, but you can values in the methods that Layer holds
func setup(handle *cudnn.Handler, mode activation.Mode, nanproped cudnn.NanMode, af, bf, ab, bb, coef float64) (*Layer, error) {

	act, err := activation.Stage(handle, mode, nanproped, coef)
	if err != nil {
		return nil, err
	}

	return &Layer{

		act: act,
		fwd: Scalars{
			Alpha: af,
			Beta:  bf,
		},
		bwd: Scalars{
			Alpha: ab,
			Beta:  bb,
		},
	}, nil
}

//Leaky returns an activation layer set to leaky
func Leaky(handle *cudnn.Handler) (*Layer, error) {
	var flg activation.ModeFlag
	return setup(handle, flg.Leaky(), defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultleakycoef)
}

//Relu returns an activation layer set to Elu
func Relu(handle *cudnn.Handler) (*Layer, error) {
	var flg activation.ModeFlag
	return setup(handle, flg.Relu(), defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultcoef)
}

//Elu returns an activation layer set to Elu
func Elu(handle *cudnn.Handler) (*Layer, error) {
	var flg activation.ModeFlag
	return setup(handle, flg.Elu(), defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultcoef)
}

//ClippedRelu returns an activation layer set to ClippedRelu
func ClippedRelu(handle *cudnn.Handler) (*Layer, error) {
	var flg activation.ModeFlag
	return setup(handle, flg.ClippedRelu(), defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultcoef)
}

//Sigmoid returns an activation layer set to Sigmoid
func Sigmoid(handle *cudnn.Handler) (*Layer, error) {
	var flg activation.ModeFlag
	return setup(handle, flg.Sigmoid(), defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultcoef)
}

//Tanh returns an activation layer set to Tanh
func Tanh(handle *cudnn.Handler) (*Layer, error) {
	var flg activation.ModeFlag
	return setup(handle, flg.Tanh(), defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultcoef)
}

//Info Returns layer info if error is not nil then values will be set to golang default
func (a *Layer) Info() (Info, error) {
	op, err := a.act.Info()
	if err != nil {
		return Info{}, err
	}
	return Info{
		Ops: op,
		Fwd: Scalars{
			Alpha: a.fwd.Alpha,
			Beta:  a.fwd.Beta,
		},
		Bwd: Scalars{
			Alpha: a.bwd.Alpha,
			Beta:  a.bwd.Beta,
		},
		OutputMemManaged: a.memmanaged,
	}, nil
}

//SetAllScalars sets all the scalars for the fwd and bwd.
//both get 2 scalaras alpha and beta and will be stored in that order.
//and will be stored in order of fwd then bwd.
func (a *Layer) SetAllScalars(fwd2bwd2 []float64) error {
	if len(fwd2bwd2) != 4 {
		return errors.New("SetAllScalars needs to have the size of 2")
	}
	a.UpDateFwdCScalars(fwd2bwd2[0], fwd2bwd2[1])
	a.UpDateBwdCScalars(fwd2bwd2[2], fwd2bwd2[3])
	return nil
}

//UpDateFwdCScalars updates the alpha and beta scalars
func (a *Layer) UpDateFwdCScalars(alpha, beta float64) {
	a.fwd.Alpha, a.fwd.Beta = alpha, beta
}

//UpDateBwdCScalars update the alpha and beta scalars
func (a *Layer) UpDateBwdCScalars(alpha, beta float64) {
	a.bwd.Alpha, a.bwd.Beta = alpha, beta
}

//ForwardProp does the forward propigation of the activation layer
func (a *Layer) ForwardProp(handle *cudnn.Handler, x, y *layers.IO) error {
	var flg activation.ModeFlag
	switch a.act.Mode() {
	case flg.Leaky():
		return a.act.FwdProp(handle, a.fwd.Alpha, x.T(), a.fwd.Beta, y.T(), nil, nil, nil)
	case flg.Threshhold():
		return a.act.FwdProp(handle, a.fwd.Alpha, x.T(), a.fwd.Beta, y.T(), a.negCoefs.T(), a.threshold.T(), a.posCoefs.T())
	case flg.PRelu():
		return a.act.FwdProp(handle, a.fwd.Alpha, x.T(), a.fwd.Beta, y.T(), a.negCoefs.T(), nil, nil)
	default:
		return a.act.FwdProp(handle, a.fwd.Alpha, x.T(), a.fwd.Beta, y.T(), nil, nil, nil)
	}
	//	return a.act.FwdProp(handle, a.fwd.Alpha, x.T(), a.fwd.Beta, y.T(), nil, nil, nil)
}

//BackProp does the backward propigation of the activation layer
func (a *Layer) BackProp(handle *cudnn.Handler, x, y *layers.IO) error {
	var flg activation.ModeFlag
	switch a.act.Mode() {
	case flg.Leaky():
		return a.act.BwdProp(handle, a.fwd.Alpha, y.T(), y.DeltaT(), x.T(), a.fwd.Beta, x.DeltaT(), nil, nil, nil, nil, nil)
	case flg.Threshhold():
		return a.act.BwdProp(handle, a.fwd.Alpha, y.T(), y.DeltaT(), x.T(), a.fwd.Beta, x.DeltaT(), a.negCoefs.T(), a.negCoefs.DeltaT(), a.threshold.T(), a.posCoefs.T(), a.posCoefs.DeltaT())
	case flg.PRelu():
		return a.act.BwdProp(handle, a.fwd.Alpha, y.T(), y.DeltaT(), x.T(), a.fwd.Beta, x.DeltaT(), a.negCoefs.T(), a.negCoefs.DeltaT(), nil, nil, nil)
	default:
		return a.act.BwdProp(handle, a.fwd.Alpha, y.T(), y.DeltaT(), x.T(), a.fwd.Beta, x.DeltaT(), nil, nil, nil, nil, nil)
	}

}

//PosCoefs If activation layer has pos coefs for threshhold activation this would be it
func (a *Layer) PosCoefs() *layers.IO {
	return a.posCoefs
}

//NegCoefs - If activation has neg coefs for prelu or threshould activation this would be it
func (a *Layer) NegCoefs() *layers.IO {
	return a.negCoefs
}

//Threshhold - If activation  threshold values for threshould activation this would be it
func (a *Layer) Threshhold() *layers.IO {
	return a.threshold
}

//Destroy destroys the cuda allocated memory for activation
func (a *Layer) Destroy() error {
	return a.act.Destroy()
}
