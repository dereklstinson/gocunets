package fcnn

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is a fcnn layer for a network
type Layer struct {
	neurons   *layers.IO
	bias      *layers.IO
	operation *gocudnn.ConvolutionD
	fwdalgo   gocudnn.ConvFwdAlgo
	bwddata   gocudnn.ConvBwdDataAlgo
	bwdfilt   gocudnn.ConvBwdFiltAlgo
}

//CreateFromInput will take the input that is given to it and along with the handle and number of neurons wanted for the layer,
// and returns a default settings layer with all the dims set to 1(except for the feature map outputs). It will also return the *layer.IO for the output of that layer
func CreateFromInput(handle *gocudnn.Handle, neurons int32, input layers.IO) (*Layer, *layers.IO, error) {
	_, _, shape, err := input.Properties()
	if err != nil {
		return nil, nil, err
	}

	if len(shape) == 4 {
		return create4dfrominput(handle, neurons, input)
	}
	if len(shape) < 4 {
		return nil, nil, errors.New("input dims should be at least 4")
	}

	return createNdfrominput(handle, neurons, input)

}
func (l *Layer) ForwardProp() error {

	return nil
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
	err3 := l.operation.DestroyDescriptor()
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
