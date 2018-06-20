//Package trainer is a package that is used for training networks.  There is not much support for this yet.
//It will have vanilla and momentum on using a device. Its hard to build any kind of trainer using cudnn.
//the optensor is kind of limited.
package trainer

import (
	"errors"

	"github.com/dereklstinson/GoCudnn"
)

//Trainer will be used for updating weights.  Right now there is only one trainer and it is momentum.
type Trainer interface {
	UpdateWeights(*gocudnn.Handle, *gocudnn.FilterD, gocudnn.Memer, *gocudnn.FilterD, gocudnn.Memer) error
	L1L2Loss() (float32, float32)
}

//Momentum is a stuct that is used for the momentum operation in updating weights.  E
type Momentum struct {
	decay1     gocudnn.CScalar
	decay2     gocudnn.CScalar
	rate       gocudnn.CScalar
	momentum   gocudnn.CScalar
	one        gocudnn.CScalar
	zero       gocudnn.CScalar
	gsum       gocudnn.Memer
	gsumD      *gocudnn.FilterD
	loss1      float32
	loss2      float32
	oneDTensor *gocudnn.TensorD
}

//L1L2Loss returns the loss that was previously recorded.
func (t *Momentum) L1L2Loss() (float32, float32) {

	return t.loss1, t.loss2
}

//SetupMomentum sets up the trainer for one and zero put the cscalar of 1 and 0 that matches the datatype there.
//(example gocudnn.CFloat(1.0) and gocudnn.CFloat(0.0).  this is a hack, but it has to be done for sanity sake. (my sanity not yours :) )
//I know of a way to fix this, but I am not able to do that right now.  That being said. Maybe the reflect package might help (idk maybe not).
//The best thing I can think of is a type switch, but I would  have to do that for every types, and I might add some more types in the GoCudnn package.
//Or I could make some training stuff in C in the GoCudnn Package.
func SetupMomentum(decay1, decay2, rate, momentum, one, zero gocudnn.CScalar) Momentum {
	return Momentum{
		decay1:   decay1,
		decay2:   decay2,
		rate:     rate,
		momentum: momentum}
}

//LoadGsum will load the gsum values
func (t *Momentum) LoadGsum(gsum gocudnn.Memer, gsumD *gocudnn.FilterD) {
	t.gsum = gsum
	t.gsumD = gsumD
}

//UpdateWeights for now is just the momentum operation.  I might have to make a new cuda library for gocudnn. I will have to check that out.
func (t *Momentum) UpdateWeights(handle *gocudnn.Handle, dwD *gocudnn.FilterD, dw gocudnn.Memer, wD *gocudnn.FilterD, w gocudnn.Memer) error {
	dtype, fmt, dims, err := dwD.GetDescriptor()
	if err != nil {
		return err
	}
	dtype1, fmt1, dims1, err1 := wD.GetDescriptor()
	if err1 != nil {
		return err1
	}
	if dtype != dtype1 {
		return errors.New("Datatypes don't match for dwD, and wD")
	}
	switch {
	case dtype != dtype1:
		return errors.New("Datatypes don't match for dwD, and wD")
	case fmt1 != fmt:
		return errors.New("TensorFormats Don't Match")
	case len(dims) != len(dims1):
		return errors.New("Length of dims don't match")
	}

	err = handle.AddTensor(dtype, t.rate, dwD.TensorD(), dw, t.momentum, t.gsumD.TensorD(), t.gsum)
	if err != nil {
		return errors.New("Check the Dim sizes of the inputs.  Also try checking the gsum dims with that too:" + err.Error())
	}

	err = handle.AddTensor(dtype, t.one, dwD.TensorD(), dw, t.one, wD.TensorD(), w)
	if err != nil {
		return err
	}
	return handle.AddTensor(dtype, t.zero, wD.TensorD(), w, t.zero, dwD.TensorD(), dw) //This should zero out dW

}
