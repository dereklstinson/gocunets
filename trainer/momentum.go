package trainer

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
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
func (t *Momentum) L1L2Loss() (float32, float32, error) {

	return float32(t.loss1), float32(t.loss2), nil
}

//SetupMomentum sets up the trainer for one and zero put the cscalar of 1 and 0 that matches the datatype there.
//(example gocudnn.CFloat(1.0) and gocudnn.CFloat(0.0).  this is a hack, but it has to be done for sanity sake. (my sanity not yours :) )
//I know of a way to fix this, but I am not able to do that right now.  That being said. Maybe the reflect package might help (idk maybe not).
//The best thing I can think of is a type switch, but I would  have to do that for every types, and I might add some more types in the GoCudnn package.
//Or I could make some training stuff in C in the GoCudnn Package.
func SetupMomentum(decay1, decay2, rate, momentum, batch float64) *Momentum {
	return &Momentum{
		decay1:   decay1,
		decay2:   decay2,
		rate:     rate,
		momentum: momentum}
}

//SetTrainingMem will load the gsum values
func (t *Momentum) SetTrainingMem(handle gocudnn.Handler, weights *layers.IO) error {
	han, ok := handle.(*gocudnn.Handle)
	if !ok {
		return errors.New("Not Correct Handle")
	}
	var err error
	t.gsum, err = weights.T().ZeroClone(han)
	return err
}
func errorappender(comment string, err error) error {
	estring := err.Error()
	return errors.New(comment + ": " + estring)

}

/*
func (t *Momentum) UpdateWeights2(handle *gocudnn.Handle, weights *layers.IO, batch float64) error {
	var err error
	err = weights.DeltaT().ScaleValues(handle, 1.0/batch)
	if err != nil {

		return errorappender("updateweights2: ScaleValues", err)
	}
	err = t.gsum.OpAdd(handle, weights.DeltaT(), weights.DeltaT(), -t.rate, 0.0, t.momentum)
	if err != nil {

		return errorappender("updateweights2: OpAdd Momentum", err)
	}

	err = weights.T().OpAdd(handle, t.gsum, t.gsum, 1.0, 0.0, 1.0)
	if err != nil {
		return errorappender("updateweights2: OpAdd Weights", err)
	}
	return weights.DeltaT().SetValues(handle, 0.0)
}
*/

//UpdateWeights for now is just the momentum operation.  I might have to make a new cuda library for gocudnn. I will have to check that out.
func (t *Momentum) UpdateWeights(handle gocudnn.Handler, weights *layers.IO, batch int) error {
	han, ok := handle.(*gocudnn.Handle)
	if !ok {
		return (errors.New("UpdateWeights: Not Correct Handle"))
	}

	var err error

	err = weights.DeltaT().ScaleValues(han, 1.0/float64(batch))
	if err != nil {

		return errorappender("updateweights: ScaleValues", err)
	}

	//gsum = weights.DeltaT()*(-t.rate)+(gsum*t.momentum)
	err = t.gsum.AddTo(han, weights.DeltaT(), -t.rate, t.momentum)
	if err != nil {
		return err
	}
	// weights.T()=weights.T()*1 +t.gsum*1
	err = weights.T().AddTo(han, t.gsum, 1.0, 1.0)

	if err != nil {
		return err
	}

	return weights.DeltaT().SetValues(han, 0.0)

}