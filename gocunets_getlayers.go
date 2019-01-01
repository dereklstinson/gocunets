package gocunets

import (
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/batchnorm"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/cnntranspose"
	"github.com/dereklstinson/GoCuNets/layers/dropout"
)

/*

type layer struct {
	name         string
	activation   *activation.Layer
	cnn          *cnn.Layer
	fcnn         *fcnn.Layer
	softmax      *softmax.Layer
	pool         *pooling.Layer
	drop         *dropout.Layer
	batch        *batchnorm.Layer
	reshape      *reshape.Layer
	cnntranspose *cnntranspose.Layer
}

*/

//BatchNorms returns the batchnorm layers in the network if nil is returned none are found
func (m *Network) BatchNorms() []*batchnorm.Layer {
	x := make([]*batchnorm.Layer, 0)
	for i := range m.layer {
		if m.layer[i].batch != nil {
			x = append(x, m.layer[i].batch)
		}
	}
	if len(x) == 0 {
		return nil
	}
	return x
}

//Dropouts returns the dropout layers in the network if nil is returned none are found
func (m *Network) Dropouts() []*dropout.Layer {
	x := make([]*dropout.Layer, 0)
	for i := range m.layer {
		if m.layer[i].drop != nil {
			x = append(x, m.layer[i].drop)
		}
	}
	if len(x) == 0 {
		return nil
	}
	return x
}

//Transposes returns the tranpose convolution layers in the network if nil is returned none are found
func (m *Network) Transposes() []*cnntranspose.Layer {
	x := make([]*cnntranspose.Layer, 0)
	for i := range m.layer {
		if m.layer[i].cnntranspose != nil {
			x = append(x, m.layer[i].cnntranspose)
		}
	}
	if len(x) == 0 {
		return nil
	}
	return x
}

//Convolutions returns the convolution layers in the network if nil is returned none are found
func (m *Network) Convolutions() []*cnn.Layer {
	x := make([]*cnn.Layer, 0)
	for i := range m.layer {
		if m.layer[i].cnn != nil {
			x = append(x, m.layer[i].cnn)
		}
	}
	if len(x) == 0 {
		return nil
	}
	return x
}

//Activations returns the activation layers in the network if nil is returned none are found
func (m *Network) Activations() []*activation.Layer {
	x := make([]*activation.Layer, 0)
	for i := range m.layer {
		if m.layer[i].activation != nil {
			x = append(x, m.layer[i].activation)
		}
	}
	if len(x) == 0 {
		return nil
	}
	return x
}
