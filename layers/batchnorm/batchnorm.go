package batchnorm

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor/batchnorm"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer the ops of a batch norm
type Layer struct {
	b   *batchnorm.Ops
	fw  abscalars
	bwp abscalars
	bwd abscalars
	eps float64
	af  float64
}
type abscalars struct {
	a float64
	b float64
}

//LayerSetup sets the layer up. I set the defaults for alpha and beta (a,b) for the forward(1,0), backward param(1,1), and backward data(1,0) that are used in cudnn.
//I am 70 percent sure that fwd and bwd data are set correctly.  I am about 25% sure bwd param is set correctly.  I will change it when I get the change
func LayerSetup(x *layers.IO, mode gocudnn.BatchNormMode, managed bool) (*Layer, error) {
	b, err := batchnorm.Stage(x.T(), mode, managed)
	fw := abscalars{
		a: 1.0,
		b: 0.0,
	}
	bwd := abscalars{
		a: 1.0,
		b: 0.0,
	}
	bwp := abscalars{
		a: 1.0,
		b: 1.0,
	}
	return &Layer{
		b:   b,
		fw:  fw,
		bwp: bwp,
		bwd: bwd,
		eps: flaot64(2e-5),
	}, err
}
func (l *Layer) ForwardProp(handle *gocudnn.Handle) {

}
func (l *Layer) SetEps(eps float64) {
	l.eps = eps
}
func (l *Layer) SetBWPAlpha(a float64) {
	l.bwp.a = a
}
func (l *Layer) SetBWPBeta(b float64) {
	l.bwp.b = b
}

func (l *Layer) SetBWDAlpha(a float64) {
	l.bwd.a = a
}
func (l *Layer) SetBWDBeta(b float64) {
	l.bwd.b = b
}

func (l *Layer) SetFWAlpha(a float64) {
	l.fw.a = a
}
func (l *Layer) SetFWBeta(b float64) {
	l.fw.b = b
}
