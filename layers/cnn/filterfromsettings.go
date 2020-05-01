package cnn

/*
import (
	"github.com/dereklstinson/gocunets/cudnn/convolution"
	"github.com/dereklstinson/gocunets/layers"
	gocudnn "github.com/dereklstinson/gocudnn"
)

//Settings is used with a json to Build a layer from a json file
type Settings struct {
	TensorFormat       gocudnn.TensorFormat    `json:"TensorFormat"`
	AlgosSet           bool                    `json:"AlgosSet"`
	ForwardAlgo        gocudnn.ConvFwdAlgo     `json:"ForwardAlgo"`
	BackwardDataAlgo   gocudnn.ConvBwdDataAlgo `json:"BackDataAlgo"`
	BackwardFilterAlgo gocudnn.ConvBwdFiltAlgo `json:"BackFiltAlgo"`
	WorkspaceSize      gocudnn.SizeT           `json:"Wspace"`
	Fastest            bool                    `json:"Fastest"`
	DataType           gocudnn.DataType        `json:"DataType"`
	FilterDims         []int32                 `json:"Dims"`
	StrideF            []int32                 `json:"StrideF"`
	ConvolutionMode    gocudnn.ConvolutionMode `json:"ConvMode"`
	StrideC            []int32                 `json:"StrideC"`
	Pad                []int32                 `json:"Pad"`
	Dilation           []int32                 `json:"Dilation"`
	MemManaged         bool                    `json:"MemManaged"`
	PremadeWeights     bool                    `json:"PremadeWeights"`
	WeightFile         string                  `json:"WeightFile"`
	Trainer            gocudnn.TrainingMode    `json:"TrainingFlag"`
}

//SetupFromSettings is used to build a cnn layer from a json file
func SetupFromSettings(s Settings) (*Layer, error) {

	ops, err := convolution.StageOperation(s.ConvolutionMode, s.DataType, s.Pad, s.StrideC, s.Dilation)
	if err != nil {
		return nil, err
	}
	w, err := layers.BuildIO(s.TensorFormat, s.DataType, s.FilterDims, s.MemManaged)
	if err != nil {
		return nil, err
	}
	b, err := buildbias(w, s.MemManaged)
	if err != nil {
		return nil, err
	}

	alpha := 1.0
	alpha2 := 1.0
	beta := 0.0
	beta2 := 1.0
	fwd := xtras{
		alpha:  alpha,
		alpha2: alpha2,
		beta:   beta,
	}
	bwdd := xtras{
		alpha:  alpha,
		alpha2: alpha2,
		beta:   beta,
	}
	bwdf := xtras{
		alpha:  alpha,
		alpha2: alpha2,
		beta:   beta2,
	}
	ops.SetAlgos(s.ForwardAlgo, s.BackwardDataAlgo, s.BackwardFilterAlgo)
	return &Layer{
		conv:       ops,
		w:          w,
		bias:       b,
		fwd:        fwd,
		bwdd:       bwdd,
		bwdf:       bwdf,
		wspacesize: s.WorkspaceSize,
	}, nil

}
*/
