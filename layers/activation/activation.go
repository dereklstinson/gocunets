package activation

import (
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is an activation layer
type Layer struct {
	aD  *gocudnn.ActivationD
	fwd xtras
	bwd xtras
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
	return handle.ActivationForward(a.aD, a.fwd.alpha, x.TensorD(), x.Mem(), a.fwd.beta, y.TensorD(), y.Mem())
}

//BackProp does the backward propigation of the activation layer
func (a *Layer) BackProp(handle *gocudnn.Handle, y, x *layers.IO) error {
	return handle.ActivationBackward(a.aD, a.bwd.alpha, y.TensorD(), y.Mem(), y.TensorD(), y.DMem(), x.TensorD(), x.Mem(), a.bwd.beta, x.TensorD(), x.DMem())
}
