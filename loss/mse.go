package loss

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/custom/xloss"
	"github.com/dereklstinson/GoCudnn/xtra"

	"github.com/dereklstinson/GoCuNets/layers"
)

//MSE is Mean Squared Error
type MSE struct {
	op   *xloss.Ops
	loss float32
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
	err := m.op.Error(h.XHandle(), x.DeltaT(), y.T(), y.DeltaT())
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

//CreateMSECalculatorGPU creates a mean squared error calculator for gpu memory
func CreateMSECalculatorGPU(handle *cudnn.Handler, managed bool) (*MSE, error) {

	var modeflag xtra.XLossModeFlag
	xloss, err := xloss.Stage(handle.XHandle(), modeflag.MSE(), managed)
	if err != nil {
		return nil, err
	}
	return &MSE{
		op: xloss,
	}, nil
}
