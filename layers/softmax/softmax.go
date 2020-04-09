package softmax

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/softmax"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is a layer that holds the algos for softmax
type Layer struct {
	s      *softmax.Ops
	alpha  float64
	beta   float64
	balpha float64
	bbeta  float64
}

//OpMultiplier sets the alpha beta scalars for the forward and backward operations
type OpMultiplier struct {
	ForwardAlpha  float64
	ForwardBeta   float64
	BackwardAlpha float64
	BackwardBeta  float64
}

//Settings contains all the info that is needed in order to perform the softmaxoutput
type Settings struct {
	Mode gocudnn.SoftMaxMode      `json:"mode,omitempty"`
	Algo gocudnn.SoftMaxAlgorithm `json:"algo,omitempty"`
}

const defaultalpha = 1.0
const defaultbeta = 0.0
const defaultbalpha = -1.0

/*
StageLogPerChannel stages the op to do a log softmax per channel.
Options are alphas and betas forward and backward
Forward prop will be:    y= alpha * softmaxforward(x,y) + beta * y
Backward will be:   x= alpha * softmaxbackward(x,y) + beta * x
Default values are:
			Forward Prop : alpha = 1.0, beta = 0.0
			Backward Prop: alpha = -1.0, beta = 0.0
*/
func StageLogPerChannel(options *OpMultiplier) *Layer {
	if options != nil {
		return &Layer{
			s:      softmax.StageLogPerChannel(),
			alpha:  options.ForwardAlpha,
			beta:   options.ForwardBeta,
			balpha: options.BackwardAlpha,
			bbeta:  options.BackwardBeta,
		}
	}
	return &Layer{
		s:      softmax.StageLogPerChannel(),
		alpha:  defaultalpha,
		beta:   defaultbeta,
		bbeta:  defaultbeta,
		balpha: defaultbalpha,
	}
}

/*
StageLogPerInstance stages the op to do a log softmax per intance.
Options are alphas and betas forward and backward
Forward prop will be:    y= alpha * softmaxforward(x,y) + beta * y
Backward will be:   x= alpha * softmaxbackward(x,y) + beta * x
Default values are:
			Forward Prop : alpha = 1.0, beta = 0.0
			Backward Prop: alpha = -1.0, beta = 0.0
*/
func StageLogPerInstance(options *OpMultiplier) *Layer {
	if options != nil {
		return &Layer{
			s:      softmax.StageLogPerInstance(),
			alpha:  options.ForwardAlpha,
			beta:   options.ForwardBeta,
			balpha: options.BackwardAlpha,
			bbeta:  options.BackwardBeta,
		}
	}
	return &Layer{
		s:      softmax.StageLogPerInstance(),
		alpha:  defaultalpha,
		beta:   defaultbeta,
		bbeta:  defaultbeta,
		balpha: defaultbalpha,
	}
}

/*
StageFastPerChannel stages the op to do a fast softmax per channel.
Options are alphas and betas forward and backward
Forward prop will be:    y= alpha * softmaxforward(x,y) + beta * y
Backward will be:   x= alpha * softmaxbackward(x,y) + beta * x
Default values are:
			Forward Prop : alpha = 1.0, beta = 0.0
			Backward Prop: alpha = -1.0, beta = 0.0
*/
func StageFastPerChannel(options *OpMultiplier) *Layer {
	if options != nil {
		return &Layer{
			s:      softmax.StageFastPerChannel(),
			alpha:  options.ForwardAlpha,
			beta:   options.ForwardBeta,
			balpha: options.BackwardAlpha,
			bbeta:  options.BackwardBeta,
		}
	}
	return &Layer{
		s:      softmax.StageFastPerChannel(),
		alpha:  defaultalpha,
		beta:   defaultbeta,
		bbeta:  defaultbeta,
		balpha: defaultbalpha,
	}
}

/*
StageFastPerInstance stages the op to do a fast softmax per instance.
Options are alphas and betas forward and backward
Forward prop will be:    y= alpha * softmaxforward(x,y) + beta * y
Backward will be:   x= alpha * softmaxbackward(x,y) + beta * x
Default values are:
			Forward Prop : alpha = 1.0, beta = 0.0
			Backward Prop: alpha = -1.0, beta = 0.0
*/
func StageFastPerInstance(options *OpMultiplier) *Layer {
	if options != nil {
		return &Layer{
			s:      softmax.StageFastPerInstance(),
			alpha:  options.ForwardAlpha,
			beta:   options.ForwardBeta,
			balpha: options.BackwardAlpha,
			bbeta:  options.BackwardBeta,
		}
	}
	return &Layer{
		s:      softmax.StageFastPerInstance(),
		alpha:  defaultalpha,
		beta:   defaultbeta,
		bbeta:  defaultbeta,
		balpha: defaultbalpha,
	}
}

/*
StageAccuratePerChannel stages the op to do a accurate softmax per channel.
Options are alphas and betas forward and backward
Forward prop will be:    y= alpha * softmaxforward(x,y) + beta * y
Backward will be:   x= alpha * softmaxbackward(x,y) + beta * x
Default values are:
			Forward Prop : alpha = 1.0, beta = 0.0
			Backward Prop: alpha = -1.0, beta = 0.0
*/
func StageAccuratePerChannel(options *OpMultiplier) *Layer {
	if options != nil {
		return &Layer{
			s:      softmax.StageAccuratePerChannel(),
			alpha:  options.ForwardAlpha,
			beta:   options.ForwardBeta,
			balpha: options.BackwardAlpha,
			bbeta:  options.BackwardBeta,
		}
	}
	return &Layer{
		s:      softmax.StageAccuratePerChannel(),
		alpha:  defaultalpha,
		beta:   defaultbeta,
		bbeta:  defaultbeta,
		balpha: defaultbalpha,
	}
}

/*
StageAccuratePerInstance stages the op to do a accurate softmax per instance.
Options are alphas and betas forward and backward
Forward prop will be:    y= alpha * softmaxforward(x,y) + beta * y
Backward will be:   x= alpha * softmaxbackward(x,y) + beta * x
Default values are:
			Forward Prop : alpha = 1.0, beta = 0.0
			Backward Prop: alpha = -1.0, beta = 0.0
*/
func StageAccuratePerInstance(options *OpMultiplier) *Layer {
	if options != nil {
		return &Layer{
			s:      softmax.StageAccuratePerInstance(),
			alpha:  options.ForwardAlpha,
			beta:   options.ForwardBeta,
			balpha: options.BackwardAlpha,
			bbeta:  options.BackwardBeta,
		}
	}
	return &Layer{
		s:      softmax.StageAccuratePerInstance(),
		alpha:  defaultalpha,
		beta:   defaultbeta,
		bbeta:  defaultbeta,
		balpha: defaultbalpha,
	}
}

//ForwardProp performs the forward propigation y is the output
func (l *Layer) ForwardProp(handle *cudnn.Handler, x, y *layers.Tensor) error {
	return l.s.ForwardProp(handle, l.alpha, x.Volume, l.beta, y.Volume)

}

//BackProp performs the backward propigation // x is the output
func (l *Layer) BackProp(handle *cudnn.Handler, dx, dy, y *layers.Tensor) error {
	return l.s.BackProp(handle, l.balpha, y.Volume, dy.Volume, l.bbeta, dx.Volume)

}

//SetForwardScalars sets the forward alpha beta scalars
func (l *Layer) SetForwardScalars(alpha, beta float64) {
	l.alpha, l.beta = alpha, beta
}

//SetBackwardScalars sets the backward alpha,beta scalars
func (l *Layer) SetBackwardScalars(alpha, beta float64) {
	l.balpha, l.bbeta = alpha, beta
}
