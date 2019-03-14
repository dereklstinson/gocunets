package loss

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/custom/xloss"
	"github.com/dereklstinson/GoCudnn/xtra"

	"github.com/dereklstinson/GoCuNets/layers"
)

//MSE is Mean Squared Error
type MSE struct {
	op          *xloss.Ops
	loss        float32
	alpha, beta float64
}

//ErrorCPU is used for backprop
func (m *MSE) ErrorCPU(generated, target []float32) []float32 {
	der := make([]float32, len(generated))
	var lossadder float32
	for i := range generated {
		backerror := generated[i] - target[i]
		der[i] = backerror
		lossadder += (backerror * backerror)
	}
	m.loss = lossadder / (2 * float32(len(target)))

	return der
}

//ErrorGPU does the error calculation y will have to contain y.T()=NetworkOutput  y.DeltaT() = target,  X returns the errors in  x.DeltaT()
func (m *MSE) ErrorGPU(h *cudnn.Handler, x, y *layers.IO) error {
	/*err := h.SyncContext()
	if err != nil {
		return err
	}
	*/
	err := h.Sync()
	if err != nil {
		return err
	}
	err = m.op.Error(h.XHandle(), x.DeltaT(), y.T(), y.DeltaT(), m.alpha, m.beta)
	if err != nil {
		return err
	}
	err = h.Sync()
	if err != nil {
		return err
	}
	m.loss = m.op.Loss()
	return err
}

//Loss returns the loss. Should be called after ErrorGPU is called
func (m *MSE) Loss() float32 {
	return m.loss
}

//SetAlphaScalars sets the alpha scalers for the forward and backward in that order in the array
func (m *MSE) SetAlphaScalars(alphas []float64) error {
	if len(alphas) != 1 {
		return errors.New("SetAllScalars needs to have the size of 1")
	}
	m.alpha = alphas[0]

	return nil
}

//SetBetaScalars sets the beta scalers for the forward and backward in that order in the array
func (m *MSE) SetBetaScalars(betas []float64) error {
	if len(betas) != 1 {
		return errors.New("SetAllScalars needs to have the size of 1")
	}
	m.beta = betas[0]
	return nil
}

//NumAlphaScalars returns the number of scalars the activation layer has both the forward and backward propigation.
func (m *MSE) NumAlphaScalars() int {
	return 1
}

//NumBetaScalars returns the number of scalars the activation layer has both the forward and backward propigation.
func (m *MSE) NumBetaScalars() int {
	return 1
}

//CreateMSECalculatorGPU creates a mean squared error calculator for gpu memory
func CreateMSECalculatorGPU(handle *cudnn.Handler, managed bool) (*MSE, error) {

	var modeflag xtra.XLossModeFlag
	xloss, err := xloss.Stage(handle.XHandle(), modeflag.MSE(), managed)
	if err != nil {
		return nil, err
	}
	return &MSE{
		op:    xloss,
		alpha: 1,
		beta:  0,
	}, nil
}
