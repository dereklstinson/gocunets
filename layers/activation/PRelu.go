package activation

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/activation"
	"github.com/dereklstinson/GoCuNets/layers"
)

//Updateable returns a true if the layer contains parameters that are updateable
func (l *Layer) Updateable() bool {
	return l.updatable
}

//PRelu returns an activation layer set to PRelu
func PRelu(handle *cudnn.Handler, channels int32, inptfrmt cudnn.TensorFormat, dtype cudnn.DataType, managedmem bool) (*Layer, error) {
	//SetupParaChan sets up a static parametric HWC takes up a ton of mem especially if adam and adagrad is the trainmode
	var aflg activation.ModeFlag
	var tflg cudnn.TensorFormatFlag

	var dims []int32
	layer, err := setup(handle, aflg.PRelu(), defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultcoef)
	if err != nil {
		return nil, err
	}
	if inptfrmt == tflg.NHWC() {
		dims = []int32{int32(1), int32(1), int32(1), channels}
	} else {
		dims = []int32{int32(1), channels, int32(1), int32(1)}
	}
	alphas, err := layers.BuildIO(inptfrmt, dtype, dims, managedmem)
	if err != nil {
		return nil, err
	}

	err = alphas.T().SetRandomNormal(.24, .26)
	if err != nil {
		return nil, err
	}

	layer.alphasbetas = alphas
	layer.memmanaged = managedmem
	layer.updatable = true
	return layer, nil
}
