package activation

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/activation"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is an activation layer
type Layer struct {
	act         *activation.Ops
	fwd         Scalars
	bwd         Scalars
	memmanaged  bool
	updatable   bool
	nanproped   gocudnn.PropagationNAN
	alphasbetas *layers.IO
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
			Alpha: af,
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
	//fmt.Println(a.fwd.alpha, a.fwd.beta)
	if a.alphasbetas == nil {
		return a.act.FwdProp(handle, a.fwd.Alpha, x.T(), a.fwd.Beta, y.T(), nil, nil)
	}
	if a.alphasbetas.DeltaT() == nil {
		return a.act.FwdProp(handle, a.fwd.Alpha, x.T(), a.fwd.Beta, y.T(), a.alphasbetas.T(), nil)
	}
	if a.alphasbetas.T() == nil {
		return a.act.FwdProp(handle, a.fwd.Alpha, x.T(), a.fwd.Beta, y.T(), nil, a.alphasbetas.DeltaT())
	}
	return a.act.FwdProp(handle, a.fwd.Alpha, x.T(), a.fwd.Beta, y.T(), a.alphasbetas.T(), a.alphasbetas.DeltaT())
}

//BackProp does the backward propigation of the activation layer
func (a *Layer) BackProp(handle *cudnn.Handler, x, y *layers.IO) error {
	//fmt.Println(a.fwd.alpha, a.fwd.beta)
	if a.alphasbetas == nil {
		return a.act.BwdProp(handle, a.fwd.Alpha, y.T(), y.DeltaT(), x.T(), a.fwd.Beta, x.DeltaT(), nil, nil)
	}
	if a.alphasbetas.DeltaT() == nil {
		return a.act.BwdProp(handle, a.fwd.Alpha, y.T(), y.DeltaT(), x.T(), a.fwd.Beta, x.DeltaT(), a.alphasbetas.T(), nil)

	}
	if a.alphasbetas.T() == nil {
		return a.act.BwdProp(handle, a.fwd.Alpha, y.T(), y.DeltaT(), x.T(), a.fwd.Beta, x.DeltaT(), nil, a.alphasbetas.DeltaT())

	}
	return a.act.BwdProp(handle, a.fwd.Alpha, y.T(), y.DeltaT(), x.T(), a.fwd.Beta, x.DeltaT(), a.alphasbetas.DeltaT(), a.alphasbetas.DeltaT())

}

//Destroy destroys the cuda allocated memory for activation
func (a *Layer) Destroy() error {
	return a.act.Destroy()
}
