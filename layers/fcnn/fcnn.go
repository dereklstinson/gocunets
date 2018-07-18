package fcnn

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor/convolution"

	"github.com/dereklstinson/GoCuNets/layers"
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
}
type xtras struct {
	alpha1 float64
	alpha2 float64
	beta   float64
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

	conv, err := convolution.Build(mode, dtype, []int32{0, 0}, []int32{1, 1}, []int32{1, 1})
	if err != nil {
		return nil, nil, err
	}

	shape[neurons] = neurons
	weights, err := layers.BuildIO(fmt, dtype, shape, managedmem)
	if err != nil {
		return nil, nil, err
	}
	bias, err := layers.BuildIO(fmt, dtype, []int32{neurons, 1, 1, 1}, managedmem)
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
}

//ForwardProp does the forward propigation
func (l *Layer) ForwardProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	err := l.conv.FwdProp(handle, l.fwd.alpha1, x.T(), l.neurons.T(), nil, l.fwd.beta, y.T())
	if err != nil {
		return err
	}

	return y.T().AddTo(handle, l.bias.T(), l.fwd.alpha1, l.fwd.beta)
}

//BackPropData does the backprop data operation
func (l *Layer) BackPropData(handle *gocudnn.Handle, x, y *layers.IO) error {
	return l.conv.BwdPropData(handle, l.bwdd.alpha1, l.neurons.T(), y.DeltaT(), nil, l.bwdd.beta, x.DeltaT())
}

//BackPropFilter does the back prop filter operation
func (l *Layer) BackPropFilter(handle *gocudnn.Handle, x, y *layers.IO) error {
	err := l.conv.BwdPropFilt(handle, l.bwdf.alpha1, x.T(), y.DeltaT(), nil, l.bwdf.beta, l.neurons.DeltaT())
	if err != nil {
		return err
	}
	return l.conv.BwdBias(handle, l.bwdf.alpha1, y.DeltaT(), l.bwdf.beta, l.bias.DeltaT())
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
