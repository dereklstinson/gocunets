package gocunets

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	//"github.com/dereklstinson/gocunets/layers"
)

func (l *Layer) inference(handle *cudnn.Handler, fwdwspace, bwddwspace *nvidia.Malloced) error {
	x, y := l.x.Tensor, l.y.Tensor
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
	if l.other != nil {
		err = l.other.Inference(handle, x, y)
		if err != nil {
			fmt.Println("Error in l.other.Inference(handle, x,y) ")
			return err
		}
		return handle.Sync()
	}

	return errors.New("Layer Not Set Up")
}
