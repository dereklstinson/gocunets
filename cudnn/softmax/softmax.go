package softmax

import (
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	"github.com/dereklstinson/GoCudnn"
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

//Flager holds the structs that hold the methods to the flags
type Flager struct {
	Algo gocudnn.SoftMaxAlgorithmFlag
	Mode gocudnn.SoftMaxModeFlag
}

//StageOperation stages the softmax operation
func StageOperation(algo gocudnn.SoftMaxAlgorithm, mode gocudnn.SoftMaxMode) *Ops {
	return &Ops{
		algo: algo,
		mode: mode,
	}
}

//DefaultOperation builds a default ops
func DefaultOperation() *Ops {
	var s gocudnn.SoftMax
	//	fmt, dtype, dims, err := input.Properties()
	//	output, err := layers.BuildIO(fmt, dtype, dims)

	return &Ops{
		algo: s.Flgs.Algo.Accurate(),
		mode: s.Flgs.Mode.Channel(),
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
func (s *Ops) ForwardProp(handle *gocudnn.Handle, alpha float64, x *tensor.Volume, beta float64, y *tensor.Volume) error {
	_, dtype, _, err := x.Properties()
	if err != nil {
		return err
	}
	a := gocudnn.CScalarByDataType(dtype, alpha)
	b := gocudnn.CScalarByDataType(dtype, beta)

	err = s.helper.Funcs.SoftMaxForward(handle, s.algo, s.mode, a, x.TD(), x.Memer(), b, y.TD(), y.Memer())
	return err
}

//BackProp performs the backward propigation
func (s *Ops) BackProp(handle *gocudnn.Handle, alpha float64, y, dy *tensor.Volume, beta float64, dx *tensor.Volume) error {
	_, dtype, _, err := dx.Properties()
	if err != nil {
		return err
	}
	a := gocudnn.CScalarByDataType(dtype, alpha)
	b := gocudnn.CScalarByDataType(dtype, beta)

	err = s.helper.Funcs.SoftMaxBackward(handle, s.algo, s.mode, a, y.TD(), y.Memer(), dy.TD(), dy.Memer(), b, dx.TD(), dx.Memer())
	return err
}
