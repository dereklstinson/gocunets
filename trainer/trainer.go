//Package trainer is a package that is used for training networks.  There is not much support for this yet.
//It will have vanilla and momentum on using a device. Its hard to build any kind of trainer using cudnn.
//the optensor is kind of limited.
package trainer

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
)

//Trainer will be used for updating weights.  Only momentum and adam are available right now
type Trainer interface {
	UpdateWeights(ctx *cudnn.Handler, dw, w *layers.Tensor, batch, counter int) error
	L1L2Loss() (float32, float32)
	SetRates(rate, dwalpha float32)
	SetDecays(l1, l2 float32)
}

//CreateTrainingMem creates trainingmem for the trainer
func CreateTrainingMem(handle *cudnn.Handler, trainer Trainer, w *layers.Tensor) error {

	switch x := trainer.(type) {
	case *Adam:
		return x.SetTrainingMem(handle, w)
	default:
		return errors.New("only adam is supported")
	}

}
