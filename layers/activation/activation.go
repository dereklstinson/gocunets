package activation

import (
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is an activation layer
type Layer struct {
	funcs gocudnn.ActivationFuncs
	aD    *gocudnn.ActivationD
	fwd   xtras
	bwd   xtras
}

//LayerSetup sets up the activation Layer
func LayerSetup(act *gocudnn.ActivationD, fwdalpha, fwdbeta, bwdalpha, bwdbeta gocudnn.CScalar) *Layer {
	return &Layer{
		aD: act,
		fwd: xtras{
			alpha: fwdalpha,
			beta:  fwdbeta,
		},
		bwd: xtras{
			alpha: bwdalpha,
			beta:  bwdbeta,
		},
	}
}

type xtras struct {
	alpha gocudnn.CScalar
	beta  gocudnn.CScalar
}

//UpDateFwdCScalars updates the alpha and beta scalars
func (a *Layer) UpDateFwdCScalars(alpha gocudnn.CScalar, beta gocudnn.CScalar) {
	a.fwd.alpha, a.fwd.beta = alpha, beta
}

//UpDateBwdCScalars update the alpha and beta scalars
func (a *Layer) UpDateBwdCScalars(alpha gocudnn.CScalar, beta gocudnn.CScalar) {
	a.bwd.alpha, a.bwd.beta = alpha, beta
}

//ForwardProp does the forward propigation of the activation layer
func (a *Layer) ForwardProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	return a.funcs.ActivationForward(handle, a.aD, a.fwd.alpha, x.Tensor().TD(), x.Tensor().Memer(), a.fwd.beta, y.Tensor().TD(), y.Tensor().Memer())
}

//BackProp does the backward propigation of the activation layer
func (a *Layer) BackProp(handle *gocudnn.Handle, y, x *layers.IO) error {
	return a.funcs.ActivationBackward(handle,
		a.aD,
		a.bwd.alpha,
		y.Tensor().TD(),
		y.Tensor().Memer(),
		y.DTensor().TD(),
		y.DTensor().Memer(),
		x.Tensor().TD(),
		x.Tensor().Memer(),
		a.bwd.beta,
		x.DTensor().TD(),
		x.DTensor().Memer())
}
