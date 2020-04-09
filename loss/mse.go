package loss

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/reduce"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	"github.com/dereklstinson/half"

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

//ErrorGPUEX y is the target values.  x is the network output. errors will be put into dx.Volume. The target values are in y.Volume
func (m *MSE) ErrorGPUEX(h *cudnn.Handler, x, dx, y *layers.Tensor) error {
	err := h.Sync()
	if err != nil {
		return err
	}
	err = m.op.Error(h.XHandle(), dx.Volume, y.Volume, x.Volume, m.alpha, m.beta)
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

//MSE2 tries to do the mse with out calling kernel outside of cudnn
type MSE2 struct {
	h           *cudnn.Handler
	half        bool
	loss        *tensor.Volume
	losscpu     []float32
	losscpuhalf []half.Float16
	wspace      *nvidia.Malloced
	indicies    *nvidia.Malloced
	//err, actual, target *layers.Tensor
	r *reduce.Ops
}

//PerformError performs the error
//PerformError satisfies the loss layer interface
func (m *MSE2) PerformError(x, dx, y, dy *layers.Tensor) error {
	err := dx.Volume.OpAdd(m.h, y.Volume, dy.Volume, 1, -1, 0)
	if err != nil {
		return err
	}
	err = y.SetValues(m.h, 0)
	if err != nil {
		return err
	}
	err = y.Volume.OpMult(m.h, dx.Volume, dx.Volume, .5, 1, 0)
	if err != nil {
		return err
	}
	err = m.r.Reduce(m.h, m.indicies, m.wspace, 1, y.Volume, 0, m.loss)
	if err != nil {
		return err
	}

	if m.losscpuhalf != nil {
		err = m.loss.FillSlice(m.h, m.losscpuhalf)
		if err != nil {
			return err
		}
		for i := range m.losscpuhalf {

			m.losscpuhalf[i] /= 2
		}

	} else if m.losscpu != nil {
		err = m.loss.FillSlice(m.h, m.losscpu)
		if err != nil {
			return err
		}
		for i := range m.losscpu {
			m.losscpu[i] /= 2
		}

	} else {
		return errors.New(" (m *MSE2) PerformError() loss array are not initiated")
	}

	return nil
}

//Inference satisfies the gocunets.LossLayer interface
func (m *MSE2) Inference(x, y *layers.Tensor) (err error) {
	return nil
}
func (m *MSE2) TestForward(x, y, dy *layers.Tensor) (err error) {
	panic("TestForward not set")
}

//GetAverageBatchLoss gets the averagebatchloss
//It also satisfies the gocunets.LossLayer interface
func (m *MSE2) GetAverageBatchLoss() float32 {
	var loss float32
	if m.losscpu != nil {
		for i := range m.losscpu {
			loss = loss + m.losscpu[i]
		}

		return loss / (float32)(len(m.losscpu))
	}
	if m.losscpuhalf != nil {
		losscpu := half.ToFloat32(m.losscpuhalf)
		for i := range losscpu {
			loss = loss + losscpu[i]
		}
		return loss / (float32)(len(losscpu))
	}
	return -999999
}

//GetBatchLoss gets the loss by batch
//It also satisfies the gocunets.LossLayer interface
func (m *MSE2) GetBatchLoss() []float32 {
	if m.losscpu != nil {
		return m.losscpu
	}
	if m.losscpuhalf != nil {
		return half.ToFloat32(m.losscpuhalf)
	}
	return nil
}

//CreateMSE2 creates a mse2 function
func CreateMSE2(h *cudnn.Handler, target *layers.Tensor) (m *MSE2, err error) {
	dtype := target.Volume.DataType()
	frmt := target.Volume.Format()
	dims := target.Volume.Dims()
	var fp16 bool
	if dtype.Half() == target.Volume.DataType() {
		fp16 = true
	}
	m = new(MSE2)

	flg := reduce.Flags
	flg.IndFlag.NoIndices()
	flg.NanProp.Propigate()
	flg.ReduceMode.Add()

	if fp16 {
		m.half = true
		m.losscpuhalf = make([]half.Float16, dims[0])
		flg.DType.Half()
		flg.IndType.Type32Bit()

	} else {
		dtype = target.Volume.DataType()
		flg.DType.Float()
		flg.IndType.Type32Bit()
		m.losscpu = make([]float32, dims[0])
	}
	reducedims := make([]int32, len(dims))
	for i := range reducedims {
		reducedims[i] = 1
	}
	reducedims[0] = dims[0]
	m.loss, err = tensor.Build(h, frmt, dtype, reducedims)
	if err != nil {
		return nil, err
	}
	m.r, err = reduce.Stage(flg.ReduceMode, flg.DType, flg.NanProp, flg.IndFlag, flg.IndType)
	if err != nil {
		return nil, err
	}
	indiciessize, err := m.r.GetIndiciesSize(h, target.Volume, m.loss)
	if err != nil {
		return nil, err
	}
	if indiciessize == 0 {
		m.indicies = nil
	} else {
		m.indicies, err = nvidia.MallocGlobal(h, indiciessize)
		if err != nil {
			return nil, err
		}
	}

	wspacesize, err := m.r.GetWorkSpaceSize(h, target.Volume, m.loss)
	if err != nil {
		return nil, err
	}
	if wspacesize == 0 {
		m.wspace = nil
	} else {
		m.wspace, err = nvidia.MallocGlobal(h, wspacesize)
		if err != nil {
			return nil, err
		}
	}

	return m, nil
}

//ErrorGPU does the error calculation y will have to contain.
//
// y = NetworkOutput
//
// dy = target
//
//dx returns the errors.
func (m *MSE) ErrorGPU(h *cudnn.Handler, dx, y, dy *layers.Tensor) error {
	/*err := h.SyncContext()
	if err != nil {
		return err
	}
	*/
	err := h.Sync()
	if err != nil {
		return err
	}
	err = m.op.Error(h.XHandle(), dx.Volume, y.Volume, dy.Volume, m.alpha, m.beta)
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

////GetTensorX returns the input tensor
//func (m *MSE2) GetTensorX() *layers.Tensor {
//	return m.actual
//}
//
////GetTensorDX returns the tensor the holds the errors for the previous layer.
//func (m *MSE2) GetTensorDX() *layers.Tensor {
//	return m.err
//}
//
////GetTensorY returns the classifier output tensor
//func (m *MSE2) GetTensorY() *layers.Tensor {
//	return m.actual
//}
//
////GetTensorDY returns the tensor that holds target values.
//func (m *MSE2) GetTensorDY() *layers.Tensor {
//	return m.target
//}
//
////SetTensorX sets the input tensor
//func (m *MSE2) SetTensorX(x *layers.Tensor) {
//	m.actual = x
//}
//
////SetTensorDX sets the tensor that holds the errors for the previous layer.
//func (m *MSE2) SetTensorDX(dx *layers.Tensor) {
//	m.err = dx
//}
//
////SetTensorY sets the output tensor for the classifier forward
//func (m *MSE2) SetTensorY(y *layers.Tensor) {
//	m.actual = y
//}
//
////SetTensorDY sets the tensor that holds target values.
//func (m *MSE2) SetTensorDY(dy *layers.Tensor) {
//	m.target = dy
//}

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
