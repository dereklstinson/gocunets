package activation

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/activation"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is an activation layer
type Layer struct {
	act        *activation.Ops
	fwd        Scalars
	bwd        Scalars
	memmanaged bool
	nanproped  gocudnn.PropagationNAN
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
const defaultnanprop = gocudnn.PropagationNAN(0) //NotPropigateNAN

//Setup takes default settings for coef (6) and NottPropNan. alpha =1 ,beta =0
//You can change the values by using the Layer methods.
//The way that alpha and beta work is this Y=(alpha *ActivationOp)+(beta*Y).
//It's best to keep the defaults of alpha and beta, but you can values in the methods that Layer holds
func Setup(mode gocudnn.ActivationMode) (*Layer, error) {
	act, err := activation.StageOperation(mode, defaultnanprop, defaultcoef)
	if err != nil {
		return nil, err
	}
	return &Layer{
		act: act,
		fwd: Scalars{
			Alpha: defaultalpha,
			Beta:  defaultbeta,
		},
		bwd: Scalars{
			Alpha: defaultalpha,
			Beta:  defaultbeta,
		},
	}, nil
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

//UpdateCoef will update the coef.  ALthough it will change the descriptor if the activation mode doesn't use the coef scalar then it won't do anything.
func (a *Layer) UpdateCoef(coef float64) error {
	amode, nanprop, _, err := a.act.Properties()
	if err != nil {
		return err
	}
	return a.act.ReStage(amode, nanprop, coef)
}

//UpdateMode will update the mode
func (a *Layer) UpdateMode(amode gocudnn.ActivationMode) error {
	_, nanprop, coef, err := a.act.Properties()
	if err != nil {
		return err
	}

	return a.act.ReStage(amode, nanprop, coef)
}

//NotPropigateNAN sets up the layer to not propigate nan values
func (a *Layer) NotPropigateNAN() error {
	if a.nanproped == gocudnn.PropagationNAN(0) {
		return nil
	}
	a.nanproped = gocudnn.PropagationNAN(0)
	return a.updatenanprop()

}

//PropigateNAN sets up the layer to propigate nan values
func (a *Layer) PropigateNAN() error {
	if a.nanproped == gocudnn.PropagationNAN(1) {
		return nil
	}
	a.nanproped = gocudnn.PropagationNAN(1)
	return a.updatenanprop()

}

func (a *Layer) updatenanprop() error {
	amode, _, coef, err := a.act.Properties()
	if err != nil {
		return err
	}
	return a.act.ReStage(amode, a.nanproped, coef)
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
func (a *Layer) ForwardProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	//fmt.Println(a.fwd.alpha, a.fwd.beta)
	return a.act.FwdProp(handle, a.fwd.Alpha, x.T(), a.fwd.Beta, y.T())
}

//BackProp does the backward propigation of the activation layer
func (a *Layer) BackProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	return a.act.BwdProp(handle,
		a.bwd.Alpha,
		y.T(),
		y.DeltaT(),
		x.T(),
		a.bwd.Beta,
		x.DeltaT())
}

//Destroy destroys the cuda allocated memory for activation
func (a *Layer) Destroy() error {
	return a.act.Destroy()
}
