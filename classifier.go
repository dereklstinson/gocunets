package gocunets

import (
	"math"

	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/fcnn"
	"github.com/dereklstinson/GoCuNets/layers/softmax"
)

//Classifier will take the outputs of a neural network and find the error of it.  To be passed back to the rest of the network.
type Classifier struct {
	sftmx *softmax.Layer
	fcnn  *fcnn.Layer
	act   *activation.Layer
}

func outputerror(desired float32, actual float32) float32 {
	return desired - actual
}

//ActivationMode are the activationmode flags
type ActivationMode int

//ActivaitonModeFlag passes ActivationMode flags
type ActivaitonModeFlag struct {
}

//SoftMax returns ActivationMode flag for softmax
func (a ActivaitonModeFlag) SoftMax() ActivationMode {
	return ActivationMode(1)
}

//Tanh returns ActivationMode flag for Tanh
func (a ActivaitonModeFlag) Tanh() ActivationMode {
	return ActivationMode(2)
}

//Logistic returns ActivationMode flag for Logistic
func (a ActivaitonModeFlag) Logistic() ActivationMode {
	return ActivationMode(3)
}

//LossMode is the flags for loss mode
type LossMode int

//LossModeFlag will return flags for LossMode
//These will be added over time.
type LossModeFlag struct {
}

func (l LossModeFlag) Huber() LossMode {
	return LossMode(1)
}
func (l LossModeFlag) Binary() LossMode {
	return LossMode(2)
}

func crossentropyloss(target, predicted float32) float32 {
	if target == 1 {
		return -float32(math.Log(float64(predicted)))
	}
	return -float32(math.Log(float64(1 - predicted)))
}

//yHat = predicted?
//y = target
