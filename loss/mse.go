package loss

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/reduce"
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//MSE is Mean Squared Error
type MSE struct {
	workspace         *gocudnn.Malloced
	reduced           *tensor.Volume
	opreduce          *reduce.Ops
	previousinputdims []int32
	gpulossholder     []float32
	indicies          *gocudnn.Malloced
}

//LossCPU is the loss for the output
func (m MSE) LossCPU(generated, target []float32) float32 {
	sumation := float32(0)
	for i := range target {
		x := target[i] - generated[i]
		sumation += (x * x)
	}
	sumation /= 2
	return sumation / float32(len(target))
}

//ErrorCPU is used for backprop
func (m *MSE) ErrorCPU(generated, target []float32) []float32 {
	der := make([]float32, len(generated))
	for i := range generated {
		der[i] = generated[i] - target[i]
	}
	return der
}

//ErrorGPU this one is kind of tricky because I usually put the answers in the dx part of layers.
//So for this the input will be x and the output y will be back propigated to the network.
func (m *MSE) ErrorGPU(h *gocudnn.Handle, x, y *layers.IO) error {
	y.T().LoadMem(x.T().Memer())                            //Puts the values of output of x into yxc
	return y.DeltaT().OpAdd(h, x.T(), x.DeltaT(), 1, -1, 0) //Takes the error of  -x.T() (output) + x.DeltaT() (target) and puts into y.DeltaT()
}

//LossGPU takes the already calculated error placed into y copies ruduces it and returns the loss of output
func (m *MSE) LossGPU(h *gocudnn.Handle, y *layers.IO) (float32, error) {
	_, _, dims, err := y.Properties()
	if err != nil {
		return 0, err
	}
	if utils.CompareInt32(m.previousinputdims, dims) == true {
		err = m.opreduce.Reduce(h, nil, m.workspace, 1.0, y.DeltaT(), 0.0, m.reduced)
		err = m.reduced.Memer().FillSlice(m.gpulossholder)
		if err != nil {
			return 0, err
		}
		return m.gpulossholder[0], nil
	}
	err = m.buildindicies(h, y.T())
	err = m.workspacebuild(h, y.T())
	if err != nil {
		return 0, err
	}
	err = m.opreduce.Reduce(h, m.indicies, m.workspace, 1.0, y.DeltaT(), 0.0, m.reduced)
	if err != nil {
		return 0, err
	}
	err = m.reduced.Memer().FillSlice(m.gpulossholder)
	if err != nil {
		return 0, err
	}
	m.gpulossholder[0] /= 2.0
	m.previousinputdims = utils.CopyDimsInt32(dims)
	return m.gpulossholder[0], nil
}

func (m *MSE) buildindicies(h *gocudnn.Handle, y *tensor.Volume) error {
	size, err := m.opreduce.GetIndiciesSize(h, y, m.reduced)

	if err != nil {
		return err
	}
	if size == 0 {
		return nil
	}
	if m.indicies != nil {

		if size < m.indicies.ByteSize() {
			return nil
		}
		x, err := m.indicies.Atributes()
		m.workspace.Free()
		if err != nil {
			return err
		}
		if x.Managed == true {
			m.indicies, err = gocudnn.UnifiedMangedGlobal(size)
			if err != nil {
				return err
			}
		} else {
			m.indicies, err = gocudnn.Malloc(size)
			if err != nil {
				return err
			}
		}

	} else {
		if y.Unified() == true {
			m.indicies, err = gocudnn.UnifiedMangedGlobal(size)
			if err != nil {
				return err
			}
		} else {
			m.indicies, err = gocudnn.Malloc(size)
			if err != nil {
				return err
			}
		}

	}
	return nil

}
func (m *MSE) workspacebuild(h *gocudnn.Handle, y *tensor.Volume) error {

	size, err := m.opreduce.GetWorkSpaceSize(h, y, m.reduced)

	if err != nil {
		return err
	}
	if size == 0 {
		return nil
	}
	if m.workspace != nil {

		if size < m.workspace.ByteSize() {
			return nil
		}
		x, err := m.workspace.Atributes()
		m.workspace.Free()
		if err != nil {
			return err
		}
		if x.Managed == true {
			m.workspace, err = gocudnn.UnifiedMangedGlobal(size)
			if err != nil {
				return err
			}
		} else {
			m.workspace, err = gocudnn.Malloc(size)
			if err != nil {
				return err
			}
		}

	} else {
		if y.Unified() == true {
			m.workspace, err = gocudnn.UnifiedMangedGlobal(size)
			if err != nil {
				return err
			}
		} else {
			m.workspace, err = gocudnn.Malloc(size)
			if err != nil {
				return err
			}
		}

	}
	return nil
}

//CreateMSECalculatorGPU creates a mean squared error calculator for gpu memory
func CreateMSECalculatorGPU(frmt gocudnn.TensorFormat, dtype gocudnn.DataType, numofdims int, managed bool) (*MSE, error) {
	dims := make([]int32, numofdims)
	for i := 0; i < numofdims; i++ {
		dims[i] = int32(1)
	}
	var oflg gocudnn.ReduceTensorOpFlag
	var nflg gocudnn.PropagationNANFlag
	var iflg gocudnn.ReduceTensorIndicesFlag
	var itflg gocudnn.IndiciesTypeFlag

	reduced, err := tensor.Build(frmt, dtype, dims, managed)
	if err != nil {
		return nil, err
	}
	op, err := reduce.Stage(oflg.Add(), dtype, nflg.PropagateNan(), iflg.NoIndices(), itflg.Type32Bit())
	if err != nil {
		return nil, err
	}
	var mse MSE
	mse.opreduce = op
	mse.reduced = reduced
	mse.gpulossholder = make([]float32, 1)
	mse.previousinputdims = []int32{-100, -1, -400, -1}

	if err != nil {
		return nil, nil
	}
	return &mse, nil
}
