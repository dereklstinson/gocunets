//Package trainer is a package that is used for training networks.  There is not much support for this yet.
//It will have vanilla and momentum on using a device. Its hard to build any kind of trainer using cudnn.
//the optensor is kind of limited.
package trainer

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCudnn"
)

//Trainer will be used for updating weights.  Right now there is only one trainer and it is momentum.
type Trainer interface {
	UpdateWeights(ctx gocudnn.Contexter, weights *layers.IO) error
	L1L2Loss() (float32, float32, error)
}

func CreateTrainingMem(ctx gocudnn.Contexter, trainer Trainer, weights *layers.IO) error {
	switch x := trainer.(type) {
	case *Adam:
		return x.SetTrainingMem(ctx, weights)
	case *eve:
		return errors.New("There is no eve ")
	case *Momentum:
		return x.SetTrainingMem(ctx, weights)
	}

	return nil
}

//eve is a troll
type eve struct {
}

func (e *eve) UpdateWeights(ctx gocudnn.Contexter, weights *layers.IO) error {
	return errors.New("update your own weights")
}
func (e *eve) L1L2Loss() (float32, float32, error) {
	return 0, 0, nil
}
