/*
Package softmax uses the softmax functions from gocudnn which is from cudnn.   Except it doesn't use any of the flags.
*/
package softmax

import (
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/gocudnn"
)

//Ops does the softmax algo
type Ops struct {
	op   *gocudnn.SoftMaxD
	algo gocudnn.SoftMaxAlgorithm
	mode gocudnn.SoftMaxMode
}

//OpInfo Contains all the needed information to build a softmax.op
type OpInfo struct {
	Algo gocudnn.SoftMaxAlgorithm `json:"Algo"`
	Mode gocudnn.SoftMaxMode      `json:"Mode"`
}

/*Log*/

//StageLogPerChannel stages the op to do a log softmax per channel.
func StageLogPerChannel() *Ops {
	var err error
	op := new(Ops)
	op.op = gocudnn.CreateSoftMaxDescriptor()
	err = op.op.Set(op.algo.Log(), op.mode.Channel())
	if err != nil {
		panic(err)
	}
	return op
}

//StageLogPerInstance stages the op to do a log softmax per instance.
func StageLogPerInstance() *Ops {
	var err error
	op := new(Ops)
	op.op = gocudnn.CreateSoftMaxDescriptor()
	err = op.op.Set(op.algo.Log(), op.mode.Instance())
	if err != nil {
		panic(err)
	}
	return op

}

/*Fast*/

//StageFastPerChannel stages the op to do a fast softmax per channel.
func StageFastPerChannel() *Ops {
	var err error
	op := new(Ops)
	op.op = gocudnn.CreateSoftMaxDescriptor()
	err = op.op.Set(op.algo.Fast(), op.mode.Channel())
	if err != nil {
		panic(err)
	}
	return op

}

//StageFastPerInstance stages the op to do a fast softmax per instance.
func StageFastPerInstance() *Ops {
	var err error
	op := new(Ops)
	op.op = gocudnn.CreateSoftMaxDescriptor()
	err = op.op.Set(op.algo.Fast(), op.mode.Instance())
	if err != nil {
		panic(err)
	}
	return op

}

/*Accurate*/

//StageAccuratePerChannel stages the op to do a accurate softmax per channel.
func StageAccuratePerChannel() *Ops {
	var err error
	op := new(Ops)
	op.op = gocudnn.CreateSoftMaxDescriptor()
	err = op.op.Set(op.algo.Accurate(), op.mode.Channel())
	if err != nil {
		panic(err)
	}
	return op

}

//StageAccuratePerInstance stages the op to do a accurate softmax per instance.
func StageAccuratePerInstance() *Ops {
	var err error
	op := new(Ops)
	op.op = gocudnn.CreateSoftMaxDescriptor()
	err = op.op.Set(op.algo.Accurate(), op.mode.Instance())
	if err != nil {
		panic(err)
	}
	return op

}

//Stage is a method for OpInfo that will return a Pointer for the staged Ops
func (i OpInfo) Stage() *Ops {
	var err error
	op := new(Ops)
	op.algo = i.Algo
	op.mode = i.Mode
	op.op = gocudnn.CreateSoftMaxDescriptor()
	err = op.op.Set(op.algo, op.mode)
	if err != nil {
		panic(err)
	}
	return op

}

//Info returns an OpInfo which contains the information to stage an softmax op
func (s *Ops) Info() OpInfo {

	return OpInfo{
		Algo: s.algo,
		Mode: s.mode,
	}
}

//ForwardProp performs the forward propigation
func (s *Ops) ForwardProp(handle *cudnn.Handler, alpha float64, x *tensor.Volume, beta float64, y *tensor.Volume) error {
	return s.op.Forward(handle.Cudnn(), alpha, x.TD(), x, beta, y.TD(), y)

}

//BackProp performs the backward propigation
func (s *Ops) BackProp(handle *cudnn.Handler, alpha float64, y, dy *tensor.Volume, beta float64, dx *tensor.Volume) error {
	if y == nil {
		panic("Y is nil")

	}
	if dy == nil {
		panic("dy is nil")
	}
	if dx == nil {
		panic("dx is nil")
	}
	return s.op.Backward(handle.Cudnn(), alpha, y.TD(), y, dy.TD(), dy, beta, dx.TD(), dx)
}
