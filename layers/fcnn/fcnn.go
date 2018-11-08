package fcnn

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/GoCuNets/cudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/trainer"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is a fcnn layer for a network
type Layer struct {
	neurons *layers.IO
	bias    *layers.IO
	conv    *convolution.Ops
	fwd     xtras
	bwdd    xtras
	bwdf    xtras
	train   trainer.Trainer
	btrain  trainer.Trainer
}

type xtras struct {
	alpha1 float64
	alpha2 float64
	beta   float64
}

func CreateFromshapeNoOut(handle *gocudnn.Handle, neurons int32, shape []int32, managedmem bool, dtype gocudnn.DataType, frmt gocudnn.TensorFormat) (*Layer, error) {

	mode := convolution.Flags().Mode.CrossCorrelation()

	if len(shape) < 4 {
		return nil, errors.New("input dims should be at least 4")
	}

	conv, err := convolution.StageOperation(mode, dtype, []int32{0, 0}, []int32{1, 1}, []int32{1, 1})
	if err != nil {
		return nil, err
	}

	shape[0] = neurons
	weights, err := layers.BuildIO(frmt, dtype, shape, managedmem)
	if err != nil {
		return nil, err
	}
	bias, err := layers.BuildIO(frmt, dtype, []int32{1, neurons, 1, 1}, managedmem)
	if err != nil {
		return nil, err
	}
	err = bias.T().SetValues(handle, 0.0)
	if err != nil {
		return nil, err
	}
	batch := shape[0]
	_, err = conv.SetBestAlgosConsideringDims4d(handle, shape, []int32{batch, neurons, 1, 1}, shape, 0, false, dtype, frmt)
	if err != nil {
		return nil, err
	}
	lyer := &Layer{
		neurons: weights,
		bias:    bias,
		conv:    conv,
		fwd: xtras{
			alpha1: 1.0,
			alpha2: 1.0,
			beta:   0.0,
		},
		bwdd: xtras{
			alpha1: 1.0,
			alpha2: 1.0,
			beta:   0.0,
		},
		bwdf: xtras{
			alpha1: 1.0,
			alpha2: 1.0,
			beta:   1.0,
		},
	}
	err = lyer.MakeRandomFromDims(shape)
	if err != nil {
		return nil, err
	}
	return lyer, nil
	/*
		return &Layer{
			neurons: weights,
			bias:    bias,
			conv:    conv,
			fwd: xtras{
				alpha1: 1.0,
				alpha2: 1.0,
				beta:   0.0,
			},
			bwdd: xtras{
				alpha1: 1.0,
				alpha2: 1.0,
				beta:   0.0,
			},
			bwdf: xtras{
				alpha1: 1.0,
				alpha2: 1.0,
				beta:   1.0,
			},
		}, output, nil
	*/
}

//CreateFromInput will take the input that is given to it and along with the handle and number of neurons wanted for the layer,
// and returns a default settings layer with all the dims set to 1(except for the feature map outputs). It will also return the *layer.IO for the output of that layer
func CreateFromInput(handle *gocudnn.Handle, neurons int32, input *layers.IO, managedmem bool) (*Layer, *layers.IO, error) {
	mode := convolution.Flags().Mode.CrossCorrelation()
	fmt, dtype, shape, err := input.Properties()
	if err != nil {
		return nil, nil, err
	}
	if len(shape) < 4 {
		return nil, nil, errors.New("input dims should be at least 4")
	}

	conv, err := convolution.StageOperation(mode, dtype, []int32{0, 0}, []int32{1, 1}, []int32{1, 1})
	if err != nil {
		return nil, nil, err
	}

	shape[0] = neurons
	weights, err := layers.BuildIO(fmt, dtype, shape, managedmem)
	if err != nil {
		return nil, nil, err
	}
	bias, err := layers.BuildIO(fmt, dtype, []int32{1, neurons, 1, 1}, managedmem)
	if err != nil {
		return nil, nil, err
	}
	err = bias.T().SetValues(handle, 0.0)
	if err != nil {
		return nil, nil, err
	}
	odims, err := conv.OutputDim(input.T(), weights.T())
	if err != nil {
		return nil, nil, err
	}
	output, err := layers.BuildIO(fmt, dtype, odims, managedmem)
	if err != nil {
		return nil, nil, err
	}
	_, err = conv.SetBestAlgosConsidering(handle, input.T(), output.T(), weights.T(), 0, false)
	if err != nil {
		return nil, nil, err
	}
	lyer := &Layer{
		neurons: weights,
		bias:    bias,
		conv:    conv,
		fwd: xtras{
			alpha1: 1.0,
			alpha2: 1.0,
			beta:   0.0,
		},
		bwdd: xtras{
			alpha1: 1.0,
			alpha2: 1.0,
			beta:   0.0,
		},
		bwdf: xtras{
			alpha1: 1.0,
			alpha2: 1.0,
			beta:   1.0,
		},
	}
	err = lyer.MakeRandomFromFanin(input)
	if err != nil {
		return nil, nil, err
	}
	return lyer, output, nil

}

//MakeRandomFromDims will take the dims and make the weights randomized
func (l *Layer) MakeRandomFromDims(dims []int32) error {

	if len(dims) < 5 {
		fanin := float64(dims[1] * dims[2] * dims[3])
		err := l.neurons.T().SetRandom(0, 1.0, fanin)
		if err != nil {
			return err
		}

	}
	if len(dims) > 4 {
		return errors.New("Not Available yet")
	}
	return nil
}

//MakeRandomFromFanin does what it says it will make the weights random considering the fanin
func (l *Layer) MakeRandomFromFanin(input *layers.IO) error {
	_, _, dims, err := input.Properties()
	if err != nil {
		return err
	}
	if len(dims) < 5 {
		fanin := float64(dims[1] * dims[2] * dims[3])
		err := l.neurons.T().SetRandom(0, 1.0, fanin)
		if err != nil {
			return err
		}

	}
	if len(dims) > 4 {
		return errors.New("Not Available yet")
	}
	return nil
}
func appenderror(message string, err error) error {
	orig := err.Error()
	return errors.New(message + ": " + orig)
}

func (l *Layer) WeightsFillSlice(input interface{}) error {
	return l.neurons.T().Memer().FillSlice(input)

}
func (l *Layer) MakeOutputTensor(batch int) (*layers.IO, error) {
	frmt, dtype, dims, err := l.neurons.Properties()
	if err != nil {
		return nil, err
	}
	managed := l.neurons.IsManaged()
	neurons := dims[0]
	var frmtflg gocudnn.TensorFormatFlag
	if frmt == frmtflg.NCHW() {
		return layers.BuildIO(frmt, dtype, []int32{int32(batch), neurons, int32(1), int32(1)}, managed)
	} else if frmt == frmtflg.NHWC() {
		return layers.BuildIO(frmt, dtype, []int32{int32(batch), int32(1), int32(1), neurons}, managed)
	}
	return nil, errors.New("Not Supported Format")
}

func (l *Layer) DeltaWeights(input interface{}) error {
	return l.neurons.DeltaT().Memer().FillSlice(input)
}

//Destroy frees all the memory associated with Layer both device and host memory (descriptors/algos)
func (l *Layer) Destroy() error {
	return destroy(l)
}
func destroy(l *Layer) error {
	var flag bool

	err1 := l.bias.Destroy()
	if err1 != nil {
		flag = true
	}
	err2 := l.neurons.Destroy()
	if err2 != nil {
		flag = true
	}
	err3 := l.conv.Destroy()
	if err3 != nil {
		flag = true
	}

	if flag == true {
		return fmt.Errorf("error:TensorD: %s,FilterD: %s,Memory: %s", err1, err2, err3)
	}
	return nil
}

//LoadTrainer sets up the momentum trainer
func (l *Layer) LoadTrainer(ctx gocudnn.Handler, trainerweights, trainerbias trainer.Trainer) error {
	var err error
	l.train = trainerweights
	err = trainer.CreateTrainingMem(ctx, l.train, l.neurons)
	if err != nil {
		return err
	}
	l.btrain = trainerbias
	err = trainer.CreateTrainingMem(ctx, l.btrain, l.bias)
	if err != nil {
		return err
	}
	return nil
}

//UpdateWeights updates the weights
func (l *Layer) UpdateWeights(ctx gocudnn.Handler, batch int) error {
	err := l.btrain.UpdateWeights(ctx, l.bias, batch)
	if err != nil {
		return err
	}
	return l.train.UpdateWeights(ctx, l.neurons, batch)
}

func dimscheck(a, b []int32) error {
	if len(a) != len(b) {
		return errors.New("num of dims not same")
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return fmt.Errorf("a[%d]=%d,b[%d]=%d", i, a[i], i, b[i])
		}
	}
	return nil

}
