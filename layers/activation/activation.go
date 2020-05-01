package activation

import (
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/activation"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/reduce"
	"github.com/dereklstinson/gocunets/layers"
	"github.com/dereklstinson/gocunets/trainer"
	gocudnn "github.com/dereklstinson/gocudnn"
)

//Layer is an activation layer
type Layer struct {
	act                          *activation.Ops
	reduce                       *reduce.Ops
	fwd                          Scalars
	bwd                          Scalars
	bwp                          Scalars
	mode                         activation.Mode
	memmanaged                   bool
	updatable                    bool
	nanproped                    gocudnn.NANProp
	threshandpreludims           []int32
	posmin, posmax               float32
	negmin, negmax               float32
	threshmin, threshmax         float32
	negcotrain                   trainer.Trainer
	poscotrain                   trainer.Trainer
	thresholdtrain               trainer.Trainer
	l1n, l2n, l1p, l2p, l1t, l2t float32
	posCoefs                     *layers.Tensor
	negCoefs                     *layers.Tensor
	threshold                    *layers.Tensor
	dposCoefs                    *layers.Tensor
	dnegCoefs                    *layers.Tensor
	dthreshold                   *layers.Tensor
	numofios                     int
}

//Info is a struct that contains the info that is needed to build the activation layer
type Info struct {
	Ops              activation.OpInfo `json:"Ops"`
	Fwd              Scalars           `json:"Fwd"`
	Bwd              Scalars           `json:"Bwd"`
	OutputMemManaged bool              `json:"OutputMemManaged"`
}

//Scalars are the scalars used in the activation operation
type Scalars struct {
	Alpha float64 `json:"A"`
	Beta  float64 `json:"B"`
}

const defaultalpha = float64(1)
const defaultbeta = float64(0)
const defaultcoef = float64(6)
const defaultnanprop = gocudnn.NANProp(0) //NotPropigateNAN
const defaultleakycoef = float64(.01)

//Setup takes default settings for coef (6) and NottPropNan. alpha =1 ,beta =0
//You can change the values by using the Layer methods.
//The way that alpha and beta work is this Y=(alpha *ActivationOp)+(beta*Y).
//It's best to keep the defaults of alpha and beta, but you can values in the methods that Layer holds
func setup(handle *cudnn.Handler, mode activation.Mode, dtype gocudnn.DataType, nanproped gocudnn.NANProp, af, bf, ab, bb, coef float64) (*Layer, error) {

	act, err := activation.Stage(handle, mode, dtype, nanproped, coef)
	if err != nil {
		return nil, err
	}

	return &Layer{
		mode: mode,
		act:  act,
		fwd: Scalars{
			Alpha: af,
			Beta:  bf,
		},
		bwd: Scalars{
			Alpha: ab,
			Beta:  bb,
		},
	}, nil
}

//ContainsWeights returns true if the layer is trainable
func (a *Layer) ContainsWeights() bool {
	flg := a.mode
	switch a.mode {
	case flg.PRelu():
		return true
	case flg.Threshhold():
		return true
	}
	return false
}

//ResetWeightsHiddenMem will reset the weights to random, and the delta storage will be set to zero.
func (a *Layer) ResetWeightsHiddenMem(h *cudnn.Handler) error {
	fanin := float64(a.posCoefs.Vol())
	var err error
	if a.posCoefs != nil {
		err = a.posCoefs.SetRandom(0, 1, fanin)
		if err != nil {
			return err
		}
		err = a.dposCoefs.SetValues(h, 0)
		if err != nil {
			return err
		}
	}
	if a.negCoefs != nil {
		err = a.negCoefs.SetRandom(0, 1, fanin)
		if err != nil {
			return err
		}
		err = a.dnegCoefs.SetValues(h, 0)
		if err != nil {
			return err
		}
	}
	if a.threshold != nil {
		err := a.threshold.SetRandom(0, 1, fanin)
		if err != nil {
			return err
		}
		err = a.dthreshold.SetValues(h, 0)
		if err != nil {
			return err
		}
	}

	return nil
}

//Leaky returns an activation layer set to leaky
func Leaky(handle *cudnn.Handler, dtype gocudnn.DataType) (*Layer, error) {
	var flg activation.Mode
	return setup(handle, flg.Leaky(), dtype, defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultleakycoef)
}

//Relu returns an activation layer set to Elu
func Relu(handle *cudnn.Handler, dtype gocudnn.DataType) (*Layer, error) {
	var flg activation.Mode
	return setup(handle, flg.Relu(), dtype, defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultcoef)
}

//Elu returns an activation layer set to Elu
func Elu(handle *cudnn.Handler, dtype gocudnn.DataType) (*Layer, error) {
	var flg activation.Mode
	return setup(handle, flg.Elu(), dtype, defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultcoef)
}

//ClippedRelu returns an activation layer set to ClippedRelu
func ClippedRelu(handle *cudnn.Handler, dtype gocudnn.DataType) (*Layer, error) {
	var flg activation.Mode
	return setup(handle, flg.ClippedRelu(), dtype, defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultcoef)
}

//Sigmoid returns an activation layer set to Sigmoid
func Sigmoid(handle *cudnn.Handler, dtype gocudnn.DataType) (*Layer, error) {
	var flg activation.Mode
	return setup(handle, flg.Sigmoid(), dtype, defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultcoef)
}

//Tanh returns an activation layer set to Tanh
func Tanh(handle *cudnn.Handler, dtype gocudnn.DataType) (*Layer, error) {
	var flg activation.Mode
	return setup(handle, flg.Tanh(), dtype, defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultcoef)
}

//Info Returns layer info if error is not nil then values will be set to golang default
func (a *Layer) Info() (Info, error) {
	op, err := a.act.Info()
	if err != nil {
		return Info{}, err
	}
	return Info{
		Ops: op,
		Fwd: Scalars{
			Alpha: a.fwd.Alpha,
			Beta:  a.fwd.Beta,
		},
		Bwd: Scalars{
			Alpha: a.bwd.Alpha,
			Beta:  a.bwd.Beta,
		},
		OutputMemManaged: a.memmanaged,
	}, nil
}

/*
//SetAlphaScalars sets the alpha scalers for the forward and backward in that order in the array
func (a *Layer) SetAlphaScalars(alphas []float64) error {
	if len(alphas) != 2 {
		return errors.New("SetAllScalars needs to have the size of 2")
	}
	a.fwd.Alpha = alphas[0]
	a.bwd.Alpha = alphas[1]
	return nil
}

//SetBetaScalars sets the beta scalers for the forward and backward in that order in the array
func (a *Layer) SetBetaScalars(betas []float64) error {
	if len(betas) != 2 {
		return errors.New("SetAllScalars needs to have the size of 2")
	}
	a.fwd.Beta = betas[0]
	a.bwd.Beta = betas[1]
	return nil
}

//NumAlphaScalars returns the number of scalars the activation layer has both the forward and backward propigation.
func (a *Layer) NumAlphaScalars() int {
	return 2
}

//NumBetaScalars returns the number of scalars the activation layer has both the forward and backward propigation.
func (a *Layer) NumBetaScalars() int {
	return 2
}
*/

//SetForwardScalars updates the alpha and beta scalars
func (a *Layer) SetForwardScalars(alpha, beta float64) {
	a.fwd.Alpha, a.fwd.Beta = alpha, beta
}

//SetBackwardScalars update the alpha and beta scalars
func (a *Layer) SetBackwardScalars(alpha, beta float64) {
	a.bwd.Alpha, a.bwd.Beta = alpha, beta
}

//SetOtherScalars sets the parameter scalars if the activation function uses it.
func (a *Layer) SetOtherScalars(alpha, beta float64) {
	a.bwp.Alpha, a.bwp.Beta = alpha, beta
}

//ForwardProp does the forward propigation of the activation layer
func (a *Layer) ForwardProp(handle *cudnn.Handler, x, y *layers.Tensor) error {
	var flg activation.Mode
	switch a.act.Mode() {
	case flg.Leaky():
		return a.act.FwdProp(handle, a.fwd.Alpha, x.Volume, a.fwd.Beta, y.Volume, nil, nil, nil)
	case flg.Threshhold():
		return a.act.FwdProp(handle, a.fwd.Alpha, x.Volume, a.fwd.Beta, y.Volume, a.negCoefs.Volume, a.threshold.Volume, a.posCoefs.Volume)
	case flg.PRelu():
		return a.act.FwdProp(handle, a.fwd.Alpha, x.Volume, a.fwd.Beta, y.Volume, a.negCoefs.Volume, nil, nil)
	default:
		return a.act.FwdProp(handle, a.fwd.Alpha, x.Volume, a.fwd.Beta, y.Volume, nil, nil, nil)
	}
	//	return a.act.FwdProp(handle, a.fwd.Alpha, x.Volume, a.fwd.Beta, y.Volume, nil, nil, nil)
}

//BackProp does the backward propigation of the activation layer
//
//In-place operation is allowed for this routine; meaning dy and dx pointers may be equal. However, this requires the corresponding tensor descriptors to be identical (particularly, the strides of the input and output must match for an in-place operation to be allowed).
//
//All tensor formats are supported for 4 and 5 dimensions, however, the best performance is obtained when the strides of yDesc and xDesc are equal and HW-packed. For more than 5 dimensions the tensors must have their spatial dimensions packed.
func (a *Layer) BackProp(handle *cudnn.Handler, x, dx, y, dy *layers.Tensor) error {
	var flg activation.Mode
	switch a.act.Mode() {
	case flg.Leaky():
		return a.act.BwdProp(handle, a.fwd.Alpha, y.Volume, dy.Volume, x.Volume, a.fwd.Beta, dx.Volume, nil, nil, nil, nil, nil, nil)
	case flg.Threshhold():
		return a.act.BwdProp(handle, a.fwd.Alpha, y.Volume, dy.Volume, x.Volume, a.fwd.Beta, dx.Volume, a.negCoefs.Volume, a.dnegCoefs.Volume, a.threshold.Volume, a.dthreshold.Volume, a.posCoefs.Volume, a.dposCoefs.Volume)
	case flg.PRelu():
		return a.act.BwdProp(handle, a.fwd.Alpha, y.Volume, dy.Volume, x.Volume, a.fwd.Beta, dx.Volume, a.negCoefs.Volume, a.dnegCoefs.Volume, nil, nil, nil, nil)
	default:
		return a.act.BwdProp(handle, a.fwd.Alpha, y.Volume, dy.Volume, x.Volume, a.fwd.Beta, dx.Volume, nil, nil, nil, nil, nil, nil)
	}

}

//PosCoefs If activation layer has pos coefs for threshhold activation this would be it
func (a *Layer) PosCoefs() *layers.Tensor {
	return a.posCoefs
}

//NegCoefs - If activation has neg coefs for prelu or threshould activation this would be it
func (a *Layer) NegCoefs() *layers.Tensor {
	return a.negCoefs
}

//Threshhold - If activation  threshold values for threshould activation this would be it
func (a *Layer) Threshhold() *layers.Tensor {
	return a.threshold
}

/*
//Destroy destroys the cuda allocated memory for activation
func (a *Layer) Destroy() error {
	return a.act.Destroy()
}
*/
