package activation

import (
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/activation"
	gocudnn "github.com/dereklstinson/gocudnn"
)

//TrainersNeeded returns the number of trainers needed
func (l *Layer) TrainersNeeded() int {
	return l.numofios
}

//PRelu returns an activation layer set to PRelu
func PRelu(handle *cudnn.Handler, dtype gocudnn.DataType, managedmem bool) (*Layer, error) {
	//SetupParaChan sets up a static parametric HWC takes up a ton of mem especially if adam and adagrad is the trainmode
	var aflg activation.Mode

	layer, err := setup(handle, aflg.PRelu(), dtype, defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultcoef)
	if err != nil {
		return nil, err
	}
	layer.memmanaged = managedmem
	layer.updatable = true
	layer.numofios = 1
	return layer, nil
}
