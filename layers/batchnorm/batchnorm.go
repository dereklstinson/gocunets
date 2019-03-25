package batchnorm

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/batchnorm"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/trainer"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

const alphaforwarddefault = 1
const betaforwarddefault = 0
const alphabackwarddefault = 1
const betabackwarddefault = 0
const alphabackwardparamdefault = 1
const betabackwardparamdefault = 1
const trainingfactoringlimit = 1000

//Layer the ops of a batch norm
type Layer struct {
	b                      *batchnorm.Ops
	fw                     abscalars
	bwp                    abscalars
	bwd                    abscalars
	bias                   *layers.IO
	scale                  *layers.IO
	eps                    float64
	af                     float64
	counter                uint64
	countermax             uint64
	mode                   gocudnn.BatchNormMode
	managed                bool
	scaletrain             trainer.Trainer
	biastrain              trainer.Trainer
	premadeweightsfromfile bool
}
type abscalars struct {
	a float64
	b float64
}

//Info contains saved info now like weights and stuff
type Info struct {
	Unified bool
	BNInfo  batchnorm.Info
	Bias    layers.Info
	Scale   layers.Info
}

//Settings contains all the paramters needed to build a batchnorm layer
type Settings struct {
	Mode    gocudnn.BatchNormMode `json:"mode,omitempty"`
	Managed bool                  `json:"managed,omitempty"`
}

//PerActivationPreset will presetup some values for the batch norm PerActivation
func PerActivationPreset(handle *cudnn.Handler) (*Layer, error) {
	//	b, err := batchnorm.PreStagePerActivation(handle, managed)
	var flg gocudnn.BatchNormMode
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
	return &Layer{

		fw:         fw,
		bwp:        bwp,
		bwd:        bwd,
		eps:        float64(1e-5),
		mode:       flg.PerActivation(),
		countermax: trainingfactoringlimit,
	}, nil
}

//SpatialPreset will presetup some values for the batch norm Spatial Mode
func SpatialPreset(handle *cudnn.Handler, managed bool) (*Layer, error) {
	//	b, err := batchnorm.PreStageSpatial(handle, managed)
	var flg gocudnn.BatchNormMode
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
	return &Layer{
		//b:    b,
		fw:         fw,
		bwp:        bwp,
		bwd:        bwd,
		eps:        float64(1e-5),
		mode:       flg.Spatial(),
		managed:    managed,
		countermax: trainingfactoringlimit,
	}, nil

}

//SpatialPersistantPreset will presetup some values for the batch norm SpatialPersistantPreset Mode
func SpatialPersistantPreset(handle *cudnn.Handler, managed bool) (*Layer, error) {
	//	b, err := batchnorm.PreStageSpatialPersistant(handle, managed)
	var flg gocudnn.BatchNormMode
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
	return &Layer{

		fw:         fw,
		bwp:        bwp,
		bwd:        bwd,
		eps:        float64(1e-5),
		mode:       flg.SpatialPersistent(),
		managed:    managed,
		countermax: trainingfactoringlimit,
	}, nil

}

//Bias returns the bias of the batch norm
func (l *Layer) Bias() *layers.IO {
	return l.bias
}

//Scale returns the scale fo the batch norm
func (l *Layer) Scale() *layers.IO {
	return l.scale
}

//Trainers returns the trainers
func (l *Layer) Trainers() (scale, bias trainer.Trainer) {
	return l.scaletrain, l.biastrain
}

//SetupPreset will allocate all the memory needed for the batch norm with the values passed when using one of the Preset functions
func (l *Layer) SetupPreset(handle *cudnn.Handler, x *layers.IO) error {

	var err error

	l.b, err = batchnorm.Stage(handle, x.T(), l.mode)
	if err != nil {
		fmt.Println("Err in stage batch norm")
		return err
	}

	frmt, dtype, dims := l.b.BiasScaleProperties()
	l.bias, err = layers.BuildIOWeights(handle, frmt, dtype, dims)
	if err != nil {
		fmt.Println("Creating Bias mem...Dims are:", dims)
		return err
	}
	l.scale, err = layers.BuildIOWeights(handle, frmt, dtype, dims)
	if err != nil {
		fmt.Println("Creating scale mem...Dims are:", dims)
		return err
	}
	err = l.scale.T().SetRandomNormal(handle, .7, 1)
	if err != nil {
		fmt.Println("error in set random normal scale", l.scale)
		return err
	}
	err = l.bias.T().SetRandomNormal(handle, .01, .1)
	if err != nil {
		fmt.Println("error in set random normal bias")
		return err
	}
	err = trainer.CreateTrainingMem(handle, l.scaletrain, l.scale)
	if err != nil {
		fmt.Println("Creating Training Mem for scale")
		return err
	}

	err = trainer.CreateTrainingMem(handle, l.biastrain, l.bias)
	if err != nil {
		fmt.Println("Creating Training Mem for bias")
		return err
	}
	l.af = 1
	return err
}

//ForwardInference Does the Testing Forward Prop and used for production
func (l *Layer) ForwardInference(handle *cudnn.Handler, x, y *layers.IO) error {

	return l.b.ForwardInference(handle, l.fw.a, l.fw.b, l.eps, x.T(), l.scale.T(), l.bias.T(), y.T())
}

//ForwardProp Does the Training Forward Prop of batch norm layer
func (l *Layer) ForwardProp(
	handle *cudnn.Handler, x, y *layers.IO) error {
	l.af = (1.0 / (1.0 + float64(l.counter)))
	err := l.b.ForwardTraining(handle, l.fw.a, l.fw.b, l.af, l.eps, x.T(), l.scale.T(), l.bias.T(), y.T())
	if l.counter < l.countermax {
		l.counter++
	}
	return err
}

//BackProp does the back propagation in training the layer
func (l *Layer) BackProp(handle *cudnn.Handler, x, y *layers.IO) error {
	return l.b.BackwardProp(handle,
		l.bwd.a,
		l.bwd.b,
		l.bwp.a,
		l.bwp.b,
		l.eps,
		x.T(),
		l.scale.T(),
		l.scale.DeltaT(),
		l.bias.DeltaT(),
		x.DeltaT(),
		y.DeltaT())
}

//UpdateWeights does the weight update
func (l *Layer) UpdateWeights(handle *cudnn.Handler, batch int) error {
	var err error

	err = l.biastrain.UpdateWeights(handle, l.bias, batch)
	if err != nil {
		return err
	}

	err = l.scaletrain.UpdateWeights(handle, l.scale, batch)
	if err != nil {
		return err
	}

	return nil
}

//LoadTrainer sets up the momentum trainer
func (l *Layer) LoadTrainer(handle *cudnn.Handler, trainerscale, trainerbias trainer.Trainer) error {

	l.scaletrain = trainerscale
	l.biastrain = trainerbias
	l.scaletrain.SetDecays(0.0, 0.0)
	l.biastrain.SetDecays(0.0, 0.0)
	return nil
}

//NumAlphaScalars returns the number of alpha scalars batch norm uses
func (l *Layer) NumAlphaScalars() int {
	return 3
}

//NumBetaScalars returns the number of beta scalars batch norm uses
func (l *Layer) NumBetaScalars() int {
	return 3
}

//SetAlphaScalars sets the forward backward and backward parameters alpha scalars in that order
func (l *Layer) SetAlphaScalars(alphas []float64) error {
	if len(alphas) != 3 {
		return errors.New("Len of alphas needs to be 3")
	}
	l.fw.a = alphas[0]
	l.bwd.a = alphas[1]
	l.bwp.a = alphas[2]
	return nil
}

//SetBetaScalars sets the forward backward and backward parameters beta scalars in that order
func (l *Layer) SetBetaScalars(betas []float64) error {
	if len(betas) != 3 {
		return errors.New("Len of betas needs to be 3")
	}

	l.fw.b = betas[0]
	l.bwd.b = betas[1]
	l.bwp.b = betas[2]
	return nil
}

//SetAllScalars sets all the scalars in this order eps 1 fwd 2, bwd 2, bwp 2.
//Each of the scalars with 2 have an alpha and beta. and they will be put in the order of alpha then beta
func (l *Layer) SetAllScalars(eps1fwd2bwd2bwp2 []float64) error {
	if len(eps1fwd2bwd2bwp2) != 7 {
		return errors.New("Length of scalars needs to be 5")
	}
	l.SetEps(eps1fwd2bwd2bwp2[0])
	l.SetFWAlpha(eps1fwd2bwd2bwp2[1])
	l.SetFWBeta(eps1fwd2bwd2bwp2[2])
	l.SetBWDAlpha(eps1fwd2bwd2bwp2[3])
	l.SetBWDBeta(eps1fwd2bwd2bwp2[4])
	l.SetBWPAlpha(eps1fwd2bwd2bwp2[5])
	l.SetBWPBeta(eps1fwd2bwd2bwp2[6])
	return nil
}

//SetEps sets epsilon
func (l *Layer) SetEps(eps float64) {
	if eps >= float64(1e-5) {
		l.eps = eps
	}

}

//SetBWPAlpha sets the alpha for bwards params
func (l *Layer) SetBWPAlpha(a float64) {
	l.bwp.a = a
}

//SetBWPBeta sets the beta for bwards params
func (l *Layer) SetBWPBeta(b float64) {
	l.bwp.b = b
}

//SetBWDAlpha sets the alpha for bwards data
func (l *Layer) SetBWDAlpha(a float64) {
	l.bwd.a = a
}

//SetBWDBeta sets beta for bwards data beta
func (l *Layer) SetBWDBeta(b float64) {
	l.bwd.b = b
}

//SetFWAlpha sets fwd alpha
func (l *Layer) SetFWAlpha(a float64) {
	l.fw.a = a
}

//SetFWBeta Sets FwdBeta
func (l *Layer) SetFWBeta(b float64) {
	l.fw.b = b
}
