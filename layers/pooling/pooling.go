package pooling

import gocudnn "github.com/dereklstinson/GoCudnn"
import "github.com/dereklstinson/GoCuNets/layers"

//Layer holds everything it needs on the pooling side in order
//to do the pooling operations.
type Layer struct {
	funcs gocudnn.PoolingFuncs
	pD    *gocudnn.PoolingD
	fwd   xtras
	bwd   xtras
}
type xtras struct {
	alpha gocudnn.CScalar
	beta  gocudnn.CScalar
}

//LayerSetup setsup the pooling layer and returns a pointer to the struct.
func LayerSetup(pD *gocudnn.PoolingD, fwdalpha, fwdbeta, bwdalpha, bwdbeta gocudnn.CScalar) *Layer {
	return &Layer{
		pD: pD,
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

//ForwardProp performs the pooling forward propigation
func (l *Layer) ForwardProp(handle *gocudnn.Handle, x, y layers.IO) error {
	return l.funcs.PoolingForward(handle, l.pD, l.fwd.alpha, x.TensorD(), x.Mem(), l.fwd.beta, y.TensorD(), y.Mem())
}

//BackProp performs the pooling backward propigation
func (l *Layer) BackProp(handle *gocudnn.Handle, x, y layers.IO) error {
	return l.funcs.PoolingBackward(handle, l.pD, l.fwd.alpha, y.TensorD(), y.Mem(), y.TensorD(), y.DMem(), x.TensorD(), x.Mem(), l.fwd.beta, x.TensorD(), x.DMem())
}
