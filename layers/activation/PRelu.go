package activation

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/activation"
)

//Updateable returns a true if the layer contains parameters that are updateable
func (l *Layer) Updateable() bool {
	return l.updatable
}

//HasWeights is kind of like Updateable, but not all activations that have weights are updateable. They might just be randomly set.
func (l *Layer) HasWeights() bool {
	if l.negCoefs != nil {
		return true
	}
	return false
}

//PRelu returns an activation layer set to PRelu
func PRelu(handle *cudnn.Handler, managedmem bool) (*Layer, error) {
	//SetupParaChan sets up a static parametric HWC takes up a ton of mem especially if adam and adagrad is the trainmode
	var aflg activation.ModeFlag

	layer, err := setup(handle, aflg.PRelu(), defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultcoef)
	if err != nil {
		return nil, err
	}
	layer.memmanaged = managedmem
	layer.updatable = true

	return layer, nil
}
