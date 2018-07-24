//Package trainer is a package that is used for training networks.  There is not much support for this yet.
//It will have vanilla and momentum on using a device. Its hard to build any kind of trainer using cudnn.
//the optensor is kind of limited.
package trainer

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCudnn"
)

//Trainer will be used for updating weights.  Right now there is only one trainer and it is momentum.
type Trainer interface {
	UpdateWeights(*gocudnn.Handle, *gocudnn.FilterD, gocudnn.Memer, *gocudnn.FilterD, gocudnn.Memer) error
	L1L2Loss() (float32, float32)
}

//Momentum is a stuct that is used for the momentum operation in updating weights.  E
type Momentum struct {
	decay1   float64
	decay2   float64
	rate     float64
	momentum float64

	gsum  *tensor.Volume
	loss1 float64
	loss2 float64
}

//L1L2Loss returns the loss that was previously recorded.
func (t *Momentum) L1L2Loss() (float64, float64) {

	return t.loss1, t.loss2
}

//SetupMomentum sets up the trainer for one and zero put the cscalar of 1 and 0 that matches the datatype there.
//(example gocudnn.CFloat(1.0) and gocudnn.CFloat(0.0).  this is a hack, but it has to be done for sanity sake. (my sanity not yours :) )
//I know of a way to fix this, but I am not able to do that right now.  That being said. Maybe the reflect package might help (idk maybe not).
//The best thing I can think of is a type switch, but I would  have to do that for every types, and I might add some more types in the GoCudnn package.
//Or I could make some training stuff in C in the GoCudnn Package.
func SetupMomentum(decay1, decay2, rate, momentum float64) Momentum {
	return Momentum{
		decay1:   decay1,
		decay2:   decay2,
		rate:     rate,
		momentum: momentum}
}

//LoadGsum will load the gsum values
func (t *Momentum) LoadGsum(handle *gocudnn.Handle, weights *layers.IO) error {
	var err error
	t.gsum, err = weights.T().ZeroClone(handle)
	return err
}

//UpdateWeights for now is just the momentum operation.  I might have to make a new cuda library for gocudnn. I will have to check that out.
func (t *Momentum) UpdateWeights(handle *gocudnn.Handle, weights *layers.IO) error {
	var err error
	err = t.gsum.AddTo(handle, weights.DeltaT(), t.rate, t.momentum)
	if err != nil {
		return err
	}
	err = weights.T().AddTo(handle, t.gsum, 1.0, 1.0)

	if err != nil {
		return err
	}

	return weights.DeltaT().SetValues(handle, 0.0)

}
