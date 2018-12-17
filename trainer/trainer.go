//Package trainer is a package that is used for training networks.  There is not much support for this yet.
//It will have vanilla and momentum on using a device. Its hard to build any kind of trainer using cudnn.
//the optensor is kind of limited.
package trainer

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
)

//Trainer will be used for updating weights.  Only momentum and adam are available right now
type Trainer interface {
	UpdateWeights(ctx *cudnn.Handler, weights *layers.IO, batch int) error
	L1L2Loss() (float32, float32)
	SetRate(rate float32)
	SetDecays(l1, l2 float32)
}

//CreateTrainingMem creates trainingmem for the trainer
func CreateTrainingMem(handle *cudnn.Handler, trainer Trainer, weights *layers.IO) error {

	switch x := trainer.(type) {
	case *Adam:
		return x.SetTrainingMem(handle, weights)
	case *Momentum:

		return x.SetTrainingMem(handle, weights)
	}

	return nil
}
