package gocunets

import (
	"errors"
	"fmt"
	//"github.com/dereklstinson/GoCuNets/layers"
)

//ForwardProp does the forward prop for a layer
func (l *Layer) forwardprop() error {
	//	return l.h.w.Work(func() error {
	fwdws := l.workspacefwd
	x, y := l.x.Tensor, l.y.Tensor
	err := l.h.Sync()
	if err != nil {
		fmt.Println("Error During First sync")
		return err
	}
	if l.cnn != nil {
		err = l.cnn.ForwardProp(l.h.Handler, fwdws, x, y)
		if err != nil {
			return err
		}
		err = l.h.Sync()
		if err != nil {
			fmt.Println("Sync Error in CNN")
		}
		return nil
	}

	if l.drop != nil {
		err = l.drop.ForwardProp(l.h.Handler, x, y)
		if err != nil {
			return err
		}
		err = l.h.Sync()
		if err != nil {
			fmt.Println("Sync Error in Drop")
		}
		return nil
	}
	if l.activation != nil {

		err = l.activation.ForwardProp(l.h.Handler, x, y)
		if err != nil {
			return err
		}
		err = l.h.Sync()
		if err != nil {
			fmt.Println("Sync Error in Activation")
		}
		return nil
	}

	if l.pool != nil {
		err = l.pool.ForwardProp(l.h.Handler, x, y)
		if err != nil {
			return err
		}
		err = l.h.Sync()
		if err != nil {
			fmt.Println("Sync Error in Pool")
		}
		return nil
	}

	if l.reshape != nil {
		err = l.reshape.ForwardProp(l.h.Handler, x, y)
		if err != nil {
			return err
		}
		err = l.h.Sync()
		if err != nil {
			fmt.Println("Sync Error in reshape")
		}
		return nil

	}
	if l.batch != nil {
		err = l.batch.ForwardProp(l.h.Handler, x, y)

		if err != nil {
			fmt.Println("Error In Batch ")
			return err
		}
		err = l.h.Sync()
		if err != nil {
			fmt.Println("Sync Error in BatchNorm")
		}
		return nil
	}
	if l.cnntranspose != nil {
		err = l.cnntranspose.ForwardProp(l.h.Handler, fwdws, x, y)
		if err != nil {
			fmt.Println("Error in Transpose ForwardProp ")
			return err
		}
		err = l.h.Sync()
		if err != nil {
			fmt.Println("Sync Error in CnnTranspose")
			return err
		}
		return nil
	}
	return errors.New("Layer Not Set Up")
	//	})

}
