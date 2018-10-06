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

//DefaultSetup takes default settings for coef (6) and NottPropNan
func DefaultSetup(input *layers.IO, mode gocudnn.ActivationMode, memmanaged bool) (*Layer, *layers.IO, error) {
	fmt, dtype, dims, err := input.Properties()
	if err != nil {
		return nil, nil, err
	}
	act, err := activation.StageOperation(mode, defaultnanprop, defaultcoef)
	if err != nil {
		return nil, nil, err
	}
	output, err := layers.BuildIO(fmt, dtype, dims, memmanaged)
	if err != nil {
		return nil, nil, err
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
		memmanaged: memmanaged,
	}, output, nil
}

//SetupNoOut takes default settings for coef (6) and NottPropNan
func SetupNoOut(input *layers.IO, mode gocudnn.ActivationMode, memmanaged bool) (*Layer, error) {

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
		memmanaged: memmanaged,
	}, nil
}

//Stage stages the layer it also needs input put layer to Build the output layer
func (i Info) Stage(input *layers.IO) (*Layer, *layers.IO, error) {
	fmt, dtype, dims, err := input.Properties()
	if err != nil {
		return nil, nil, err
	}
	if i.Ops.Coef == 0 {
		i.Ops.Coef = defaultcoef
	}
	op, err := i.Ops.Stage()
	if err != nil {
		return nil, nil, err
	}
	output, err := layers.BuildIO(fmt, dtype, dims, i.OutputMemManaged)
	if err != nil {
		return nil, nil, err
	}
	if i.Fwd.Alpha == 0 && i.Fwd.Beta == 0 {
		i.Fwd.Alpha = defaultalpha
	}
	if i.Bwd.Alpha == 0 && i.Bwd.Beta == 0 {
		i.Bwd.Alpha = defaultalpha
	}
	return &Layer{
		act:        op,
		fwd:        i.Fwd,
		bwd:        i.Bwd,
		memmanaged: i.OutputMemManaged,
	}, output, nil

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

//LayerSetup sets up the activation Layer
func LayerSetup(input *layers.IO, mode gocudnn.ActivationMode, NanProp gocudnn.PropagationNAN, memmanaged bool) (*Layer, *layers.IO, error) {
	fmt, dtype, dims, err := input.Properties()
	if err != nil {
		return nil, nil, err
	}
	act, err := activation.StageOperation(mode, NanProp, defaultcoef)
	if err != nil {
		return nil, nil, err
	}
	output, err := layers.BuildIO(fmt, dtype, dims, memmanaged)
	if err != nil {
		return nil, nil, err
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
		memmanaged: memmanaged,
	}, output, nil
}

//UpdateCoef will update the coef
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

//UpdateNanProp will update the nanprop
func (a *Layer) UpdateNanProp(nanprop gocudnn.PropagationNAN) error {
	amode, _, coef, err := a.act.Properties()
	if err != nil {
		return err
	}

	return a.act.ReStage(amode, nanprop, coef)
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
