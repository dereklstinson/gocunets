package trainer

import (
	"errors"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/tensor"
	"github.com/dereklstinson/gocunets/layers"
)

//Momentum is a stuct that is used for the momentum operation in updating weights.  E
type Momentum struct {
	decay1   float64
	decay2   float64
	rate     float64
	momentum float64
	gsum     *tensor.Volume
	loss1    float64
	loss2    float64
}

//L1L2Loss returns the loss that was previously recorded.
func (t *Momentum) L1L2Loss() (float32, float32) {

	return float32(t.loss1), float32(t.loss2)
}

//SetDecays sets the decay rates for the trainer
func (t *Momentum) SetDecays(l1, l2 float32) {
	t.decay1, t.decay2 = float64(l1), float64(l2)
}

//SetupMomentum sets up the trainer for one and zero put the cscalar of 1 and 0 that matches the datatype there.
//(example gocudnn.CFloat(1.0) and gocudnn.CFloat(0.0).  this is a hack, but it has to be done for sanity sake. (my sanity not yours :) )
//I know of a way to fix this, but I am not able to do that right now.  That being said. Maybe the reflect package might help (idk maybe not).
//The best thing I can think of is a type switch, but I would  have to do that for every types, and I might add some more types in the gocudnn package.
//Or I could make some training stuff in C in the gocudnn Package.
func SetupMomentum(decay1, decay2, rate, momentum, batch float64) *Momentum {
	return &Momentum{
		decay1:   decay1,
		decay2:   decay2,
		rate:     rate,
		momentum: momentum}
}

//SetRate the Learning Rate of momentum
func (t *Momentum) SetRate(rate float32) {
	t.rate = float64(rate)
}

//SetTrainingMem will load the gsum values
func (t *Momentum) SetTrainingMem(handle *cudnn.Handler, w *layers.Tensor) error {

	var err error
	t.gsum, err = tensor.ZeroClone(handle, w.Volume)
	return err
}
func errorappender(comment string, err error) error {
	estring := err.Error()
	return errors.New(comment + ": " + estring)

}

//UpdateWeights for now is just the momentum operation.  I might have to make a new cuda library for gocudnn. I will have to check that out.
func (t *Momentum) UpdateWeights(handle *cudnn.Handler, dw, w *layers.Tensor, batch int) error {

	var err error

	err = dw.ScaleValues(handle, 1.0/float64(batch))
	if err != nil {

		return errorappender("updateweights: ScaleValues", err)
	}

	//gsum = weights.DeltaT()*(-t.rate)+(gsum*t.momentum)
	err = t.gsum.AddTo(handle, dw.Volume, -t.rate, t.momentum)
	if err != nil {
		return err
	}
	// weights.T()=weights.T()*1 +t.gsum*1
	err = w.AddTo(handle, t.gsum, 1.0, 1.0)

	if err != nil {
		return err
	}

	return dw.SetValues(handle, 0.0)

}
