package gocunets

import (
	"math"

	"github.com/dereklstinson/gocunets/layers/activation"
	"github.com/dereklstinson/gocunets/layers/softmax"
)

//Classifier will take the outputs of a neural network and find the error of it.  To be passed back to the rest of the network.
type Classifier struct {
	sftmx *softmax.Layer
	act   *activation.Layer
}

func outputerror(desired float32, actual float32) float32 {
	return desired - actual
}

//ActMode are the activationmode flags
type ActMode int

//ActivaitonModeFlag passes ActivationMode flags
type ActivaitonModeFlag struct {
}

//SoftMax returns ActivationMode flag for softmax
func (a ActivaitonModeFlag) SoftMax() ActMode {
	return ActMode(1)
}

//Tanh returns ActivationMode flag for Tanh
func (a ActivaitonModeFlag) Tanh() ActMode {
	return ActMode(2)
}

//Logistic returns ActivationMode flag for Logistic
func (a ActivaitonModeFlag) Logistic() ActMode {
	return ActMode(3)
}

//LossMode is the flags for loss mode
type LossMode int

//LossModeFlag will return flags for LossMode
//These will be added over time.
type LossModeFlag struct {
}

//Huber is a loss mode
func (l LossModeFlag) Huber() LossMode {
	return LossMode(1)
}

//Binary is a loss mode
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
