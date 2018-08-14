package activation

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor/activation"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is an activation layer
type Layer struct {
	act *activation.Ops
	fwd xtras
	bwd xtras
}
type Info struct {
	Ops activation.Info `json:Ops`
	Fwd Scalars         `json:"Fwd"`
	Bwd Scalars         `json:"Bwd"`
}
type Scalars struct {
	Alpha float64 `json:"A"`
	Beta  float64 `json:"B"`
}

//LayerSetup sets up the activation Layer
func LayerSetup(input *layers.IO, mode gocudnn.ActivationMode, NanProp gocudnn.PropagationNAN, coef float64, fwdalpha, fwdbeta, bwdalpha, bwdbeta float64, memmanaged bool) (*Layer, *layers.IO, error) {
	fmt, dtype, dims, err := input.Properties()
	act, err := activation.Build(mode, NanProp, coef)
	if err != nil {
		return nil, nil, err
	}
	output, err := layers.BuildIO(fmt, dtype, dims, memmanaged)
	if err != nil {
		return nil, nil, err
	}
	return &Layer{
		act: act,
		fwd: xtras{
			alpha: fwdalpha,
			beta:  fwdbeta,
		},
		bwd: xtras{
			alpha: bwdalpha,
			beta:  bwdbeta,
		},
	}, output, nil
}

type xtras struct {
	alpha float64
	beta  float64
}

//UpDateFwdCScalars updates the alpha and beta scalars
func (a *Layer) UpDateFwdCScalars(alpha, beta float64) {
	a.fwd.alpha, a.fwd.beta = alpha, beta
}

//UpDateBwdCScalars update the alpha and beta scalars
func (a *Layer) UpDateBwdCScalars(alpha, beta float64) {
	a.bwd.alpha, a.bwd.beta = alpha, beta
}

//ForwardProp does the forward propigation of the activation layer
func (a *Layer) ForwardProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	//fmt.Println(a.fwd.alpha, a.fwd.beta)
	return a.act.FwdProp(handle, a.fwd.alpha, x.T(), a.fwd.beta, y.T())
}

//BackProp does the backward propigation of the activation layer
func (a *Layer) BackProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	return a.act.BwdProp(handle,
		a.bwd.alpha,
		y.T(),
		y.DeltaT(),
		x.T(),
		a.bwd.beta,
		x.DeltaT())
}

//Destroy destroys the cuda allocated memory for activation
func (a *Layer) Destroy() error {
	return a.act.Destroy()
}
