package batchnorm

import (
	"fmt"

	gocudnn "github.com/dereklstinson/gocudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/batchnorm"
	"github.com/dereklstinson/gocunets/layers"
)

const alphaforwarddefault = 1
const betaforwarddefault = 0
const alphabackwarddefault = 1
const betabackwarddefault = 0
const alphabackwardparamdefault = 1
const betabackwardparamdefault = 1
const trainingfactoringlimit = 128

//Layer the ops of a batch norm
type Layer struct {
	b                      *batchnorm.Ops
	fw                     abscalars
	bwp                    abscalars
	bwd                    abscalars
	bias                   *layers.Tensor
	scale                  *layers.Tensor
	dbias                  *layers.Tensor
	dscale                 *layers.Tensor
	eps                    float64
	af                     float64
	counter                uint64
	countermax             uint64
	managed                bool
	premadeweightsfromfile bool
}
type abscalars struct {
	a float64
	b float64
}

/*
//Info contains saved info now like weights and stuff
type Info struct {
	Unified bool
	BNInfo  batchnorm.Info
	Bias    layers.Info
	Scale   layers.Info
}
*/

//Settings contains all the paramters needed to build a batchnorm layer
type Settings struct {
	Mode    gocudnn.BatchNormMode `json:"mode,omitempty"`
	Managed bool                  `json:"managed,omitempty"`
}

//PerActivationPreset will presetup some values for the batch norm PerActivation
func PerActivationPreset(handle *cudnn.Handler) (*Layer, error) {
	//	b, err := batchnorm.PreStagePerActivation(handle, managed)

	fw := abscalars{
		a: alphaforwarddefault,
		b: betaforwarddefault,
	}
	bwd := abscalars{
		a: alphabackwarddefault,
		b: betabackwarddefault,
	}
	bwp := abscalars{
		a: alphabackwardparamdefault,
		b: betabackwardparamdefault,
	}
	op, err := batchnorm.PreStagePerActivation(handle)
	if err != nil {
		return nil, err
	}
	return &Layer{
		b:          op,
		fw:         fw,
		bwp:        bwp,
		bwd:        bwd,
		eps:        float64(1e-5),
		countermax: trainingfactoringlimit,
	}, nil
}

//SpatialPreset will presetup some values for the batch norm Spatial Mode
func SpatialPreset(handle *cudnn.Handler) (*Layer, error) {
	//	b, err := batchnorm.PreStageSpatial(handle, managed)

	fw := abscalars{
		a: alphaforwarddefault,
		b: betaforwarddefault,
	}
	bwd := abscalars{
		a: alphabackwarddefault,
		b: betabackwarddefault,
	}
	bwp := abscalars{
		a: alphabackwardparamdefault,
		b: betabackwardparamdefault,
	}
	op, err := batchnorm.PreStageSpatial(handle)
	if err != nil {
		return nil, err
	}
	return &Layer{
		b:   op,
		fw:  fw,
		bwp: bwp,
		bwd: bwd,
		eps: float64(1e-5),

		countermax: trainingfactoringlimit,
	}, nil

}

//SpatialPersistantPreset will presetup some values for the batch norm SpatialPersistantPreset Mode
func SpatialPersistantPreset(handle *cudnn.Handler) (*Layer, error) {
	//	b, err := batchnorm.PreStageSpatialPersistant(handle, managed)

	fw := abscalars{
		a: alphaforwarddefault,
		b: betaforwarddefault,
	}
	bwd := abscalars{
		a: alphabackwarddefault,
		b: betabackwarddefault,
	}
	bwp := abscalars{
		a: alphabackwardparamdefault,
		b: betabackwardparamdefault,
	}
	op, err := batchnorm.PreStageSpatialPersistant(handle)
	if err != nil {
		return nil, err
	}
	return &Layer{
		b:          op,
		fw:         fw,
		bwp:        bwp,
		bwd:        bwd,
		eps:        float64(1e-5),
		countermax: trainingfactoringlimit,
	}, nil

}

//Bias returns the bias of the batch norm
func (l *Layer) Bias() *layers.Tensor {
	return l.bias
}

//Scale returns the scale fo the batch norm
func (l *Layer) Scale() *layers.Tensor {
	return l.scale
}

//SetupPreset will allocate all the memory needed for the batch norm with the values passed when using one of the Preset functions
func (l *Layer) SetupPreset(handle *cudnn.Handler, x *layers.Tensor) (err error) {

	err = l.b.Stage(handle, x.Volume)
	if err != nil {
		fmt.Println("Err in stage batch norm")
		return err
	}
	frmt, dtype, dims := l.b.BiasScaleProperties()
	l.bias, err = layers.CreateTensor(handle, frmt, dtype, dims)
	if err != nil {
		fmt.Println("Creating Bias mem...Dims are:", dims)
		return err
	}
	l.scale, err = layers.CreateTensor(handle, frmt, dtype, dims)
	if err != nil {
		fmt.Println("Creating scale mem...Dims are:", dims)
		return err
	}
	err = l.scale.Volume.SetRandomNormal(handle, .7, 1)
	if err != nil {
		fmt.Println("error in set random normal scale", l.scale)
		return err
	}
	err = l.bias.Volume.SetRandomNormal(handle, .01, .1)
	if err != nil {
		fmt.Println("error in set random normal bias")
		return err
	}

	l.af = 1
	return err
}

//ForwardInference Does the Testing Forward Prop and used for production
func (l *Layer) ForwardInference(handle *cudnn.Handler, x, y *layers.Tensor) error {

	return l.b.ForwardInference(handle, l.fw.a, l.fw.b, l.eps, x.Volume, l.scale.Volume, l.bias.Volume, y.Volume)
}

//ForwardProp Does the Training Forward Prop of batch norm layer
func (l *Layer) ForwardProp(
	handle *cudnn.Handler, x, y *layers.Tensor) error {

	l.af = (1.0 / (1.0 + float64(l.counter)))
	err := l.b.ForwardTraining(handle, l.fw.a, l.fw.b, l.af, l.eps, x.Volume, l.scale.Volume, l.bias.Volume, y.Volume)
	if l.counter < l.countermax {
		l.counter++
	}

	return err
}

//BackProp does the back propagation in training the layer
func (l *Layer) BackProp(handle *cudnn.Handler, x, dx, dy *layers.Tensor) error {
	return l.b.BackwardProp(handle,
		l.bwd.a,
		l.bwd.b,
		l.bwp.a,
		l.bwp.b,
		l.eps,
		x.Volume,
		l.scale.Volume,
		l.dscale.Volume,
		l.dbias.Volume,
		dx.Volume,
		dy.Volume)
}

//GetWeights gets the weights of the batchnorm operation
func (l *Layer) GetWeights() []*layers.Tensor {
	return []*layers.Tensor{l.scale, l.bias}
}

//GetDeltaWeights gets the deltaweights of the batchnorm operation
func (l *Layer) GetDeltaWeights() []*layers.Tensor {
	return []*layers.Tensor{l.dscale, l.dbias}
}

//SetForwardScalars sets the forward scalars
func (l *Layer) SetForwardScalars(alpha, beta float64) {
	l.fw.a, l.fw.b = alpha, beta
}

//SetBackwardScalars sets the backward scalars
func (l *Layer) SetBackwardScalars(alpha, beta float64) {
	l.bwd.a, l.bwd.b = alpha, beta
}

//SetOtherScalars these set the weights
func (l *Layer) SetOtherScalars(alpha, beta float64) {
	l.bwp.a, l.bwp.b = alpha, beta
}

//SetEps sets epsilon
func (l *Layer) SetEps(eps float64) {
	if eps >= float64(1e-5) {
		l.eps = eps
	}

}
