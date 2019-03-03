/*
Package softmax uses the softmax functions from GoCudnn which is from cudnn.   Except it doesn't use any of the flags.
*/
package softmax

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Ops does the softmax algo
type Ops struct {
	algo   gocudnn.SoftMaxAlgorithm
	helper gocudnn.SoftMax
	mode   gocudnn.SoftMaxMode
}

//OpInfo Contains all the needed information to build a softmax.op
type OpInfo struct {
	Algo gocudnn.SoftMaxAlgorithm `json:"Algo"`
	Mode gocudnn.SoftMaxMode      `json:"Mode"`
}

/*Log*/

//StageLogPerChannel stages the op to do a log softmax per channel.
func StageLogPerChannel() *Ops {
	var s gocudnn.SoftMax
	return &Ops{
		algo: s.Flgs.Algo.Log(),
		mode: s.Flgs.Mode.Channel(),
	}
}

//StageLogPerInstance stages the op to do a log softmax per instance.
func StageLogPerInstance() *Ops {
	var s gocudnn.SoftMax
	return &Ops{
		algo: s.Flgs.Algo.Log(),
		mode: s.Flgs.Mode.Instance(),
	}
}

/*Fast*/

//StageFastPerChannel stages the op to do a fast softmax per channel.
func StageFastPerChannel() *Ops {
	var s gocudnn.SoftMax
	return &Ops{
		algo: s.Flgs.Algo.Fast(),
		mode: s.Flgs.Mode.Channel(),
	}
}

//StageFastPerInstance stages the op to do a fast softmax per instance.
func StageFastPerInstance() *Ops {
	var s gocudnn.SoftMax
	return &Ops{
		algo: s.Flgs.Algo.Fast(),
		mode: s.Flgs.Mode.Instance(),
	}
}

/*Accurate*/

//StageAccuratePerChannel stages the op to do a accurate softmax per channel.
func StageAccuratePerChannel() *Ops {
	var s gocudnn.SoftMax
	return &Ops{
		algo: s.Flgs.Algo.Accurate(),
		mode: s.Flgs.Mode.Channel(),
	}
}

//StageAccuratePerInstance stages the op to do a accurate softmax per instance.
func StageAccuratePerInstance() *Ops {
	var s gocudnn.SoftMax
	return &Ops{
		algo: s.Flgs.Algo.Accurate(),
		mode: s.Flgs.Mode.Instance(),
	}
}

//Stage is a method for OpInfo that will return a Pointer for the staged Ops
func (i OpInfo) Stage() *Ops {
	return &Ops{
		algo: i.Algo,
		mode: i.Mode,
	}
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

	return s.helper.Funcs.SoftMaxForward(handle.Cudnn(), s.algo, s.mode, alpha, x.TD(), x.Memer(), beta, y.TD(), y.Memer())
}

//BackProp performs the backward propigation
func (s *Ops) BackProp(handle *cudnn.Handler, alpha float64, y, dy *tensor.Volume, beta float64, dx *tensor.Volume) error {

	return s.helper.Funcs.SoftMaxBackward(handle.Cudnn(), s.algo, s.mode, alpha, y.TD(), y.Memer(), dy.TD(), dy.Memer(), beta, dx.TD(), dx.Memer())
}
