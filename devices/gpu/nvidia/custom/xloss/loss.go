package xloss

import (
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/tensor"
	"github.com/dereklstinson/gocudnn/xtra"
)

//Ops contains the operation to do the loss function
type Ops struct {
	desc *xtra.XLossD
	loss float32
}

//Stage stages the loss operation
func Stage(handle *xtra.Handle, mode xtra.XLossMode, managed bool) (*Ops, error) {

	xloss, err := xtra.NewLossDescriptor(handle, mode)
	if err != nil {
		return nil, err
	}
	return &Ops{
		desc: xloss,
	}, nil
}

//Error calculates the MSE error dx will get the errors y is the target values dy is the network output
func (o *Ops) Error(handle *xtra.Handle, dx, y, dy *tensor.Volume, alpha, beta float64) error {
	var err error
	o.loss, err = o.desc.CalculateErrorAndLoss(handle, dx.TD(), dx, y.TD(), y, dy.TD(), dy, alpha, beta)
	return err
}

//Loss returns the loss function found. Should be called after the error is found
func (o *Ops) Loss() float32 {
	return o.loss
}
