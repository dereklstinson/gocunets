package gocunets

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
)

func (l *layer) inference(handle *cudnn.Handler, fwdwspace, bwddwspace *nvidia.Malloced, x, y *layers.IO) error {
	err := handle.Sync()
	if err != nil {
		fmt.Println("Error During First sync")
		return err
	}
	if l.cnn != nil {
		err = l.cnn.ForwardProp(handle, fwdwspace, x, y)
		if err != nil {
			return err
		}
		err = handle.Sync()
		if err != nil {
			fmt.Println("Sync Error in CNN")
		}
		return nil
	}

	if l.drop != nil {
		err = l.drop.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		err = handle.Sync()
		if err != nil {
			fmt.Println("Sync Error in Drop")
		}
		return nil
	}
	if l.activation != nil {

		err = l.activation.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		err = handle.Sync()
		if err != nil {
			fmt.Println("Sync Error in Activation")
		}
		return nil
	}
	if l.softmax != nil {
		err = l.softmax.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		err = handle.Sync()
		if err != nil {
			fmt.Println("Sync Error in Softmax")
		}
		return nil
	}
	if l.pool != nil {
		err = l.pool.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		err = handle.Sync()
		if err != nil {
			fmt.Println("Sync Error in Pool")
		}
		return nil
	}

	if l.reshape != nil {
		err = l.reshape.ForwardProp(handle, x, y)
		if err != nil {
			return err
		}
		err = handle.Sync()
		if err != nil {
			fmt.Println("Sync Error in reshape")
		}
		return nil

	}
	if l.batch != nil {
		err = l.batch.ForwardProp(handle, x, y)

		if err != nil {
			fmt.Println("Error In Batch ")
			return err
		}
		err = handle.Sync()
		if err != nil {
			fmt.Println("Sync Error in BatchNorm")
		}
		return nil
	}
	if l.cnntranspose != nil {
		err = l.cnntranspose.ForwardProp(handle, bwddwspace, x, y)
		if err != nil {
			fmt.Println("Error in Transpose ForwardProp ")
			return err
		}
		err = handle.Sync()
		if err != nil {
			fmt.Println("Sync Error in CnnTranspose")
			return err
		}
		return nil
	}
	return errors.New("Layer Not Set Up")
}

//Must run getoutput(training) before running this
func (l *layer) getoutputinference(handle *cudnn.Handler, input *layers.IO) (*layers.IO, error) {

	if l.cnn != nil {
		io, err := l.cnn.MakeOutputTensorInference(handle, input)
		if err != nil {
			fmt.Println("Error in CNN Make Output Tensor input is:", input)
		}
		return io, err
	}

	if l.pool != nil {
		return l.pool.MakeOutputLayerInference(handle, input)
	}
	if l.drop != nil {

		return input.ZeroCloneInference(handle)
	}

	if l.activation != nil {
		io, err := input.ZeroCloneInference(handle)
		if err != nil {
			fmt.Println("Error in activation Make Output Tensor input is:", input)
		}
		return io, err

	}
	if l.batch != nil {

		io, err := input.ZeroCloneInference(handle)
		if err != nil {
			fmt.Println("Error in batch Make Output Tensor input is:", input)
		}
		return io, err
	}

	if l.softmax != nil {
		return input.ZeroCloneInference(handle)
	}
	if l.reshape != nil {
		return l.reshape.MakeOutputTensorInference(handle, input)
	}
	if l.cnntranspose != nil {
		io, err := l.cnntranspose.MakeOutputTensor(handle, input)
		if err != nil {
			fmt.Println("Error in cnntranspose Make Output Tensor input is:", input)
		}
		return io, err
	}
	return nil, errors.New("Layer Needs Support")
}
