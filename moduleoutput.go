package gocunets

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia"
	"github.com/dereklstinson/gocunets/trainer"
	gocudnn "github.com/dereklstinson/gocudnn"
)

//OutputModule is just a single single convolution before it goes into the loss function
type OutputModule struct {
	id        int64
	b         *Builder
	op        *Layer
	batchsize int
}

//ID satisfies module interface
func (m *OutputModule) ID() int64 {
	return m.id
}
func xgeqy(x, y []int32, fmt gocudnn.TensorFormat) bool {
	flg := fmt
	xspace := make([]int32, len(x)-2)
	yspace := make([]int32, len(y)-2)
	switch fmt {
	case flg.NCHW():
		copy(xspace, x[2:])
		copy(yspace, y[2:])
	case flg.NHWC():
		copy(xspace, x[1:len(xspace)-1])
		copy(yspace, y[1:len(xspace)-1])
	}

	var adder int32
	for i := range xspace {
		adder += xspace[i] - yspace[i]
	}
	if adder >= 0 {
		return true
	}
	return false
}

//CreateOutputModule creates an output module
func CreateOutputModule(id int64, bldr *Builder, batch int32, fdims, pad, stride, dilation []int32, balpha, bbeta, falpha, fbeta float64) (m *OutputModule, err error) {
	m = new(OutputModule)
	m.b = bldr
	m.id = id

	m.batchsize = int(batch)

	w, dw, b, db, err := m.b.CreateConvolutionWeights(fdims)
	if err != nil {
		return nil, err
	}
	m.op, err = m.b.ConvolutionLayer(id, 1, w, dw, b, db, pad, stride, dilation)
	if err != nil {
		return nil, err
	}
	m.op.SetBackwardScalars(balpha, bbeta)
	m.op.SetOtherScalars(1, 0)
	m.op.SetForwardScalars(falpha, fbeta)

	return m, nil
}

//InitHiddenLayers will init the hidden operation
func (m *OutputModule) InitHiddenLayers(rate, decay1, decay2 float32) (err error) {

	if m.op.cnn != nil {
		err := m.op.cnn.MakeRandom(m.op.h.Handler, m.op.x.Dims())
		if err != nil {
			return err
		}

	} else if m.op.cnntranspose != nil {
		err := m.op.cnntranspose.MakeRandom(m.op.h.Handler, m.op.x.Dims())
		if err != nil {
			return err
		}

	} else {
		return errors.New("(m *OutputModule)InitHiddenLayers. CreateModule needs to be ran first")
	}
	err = m.b.h.Sync()
	if err != nil {
		return err
	}
	w, bias, err := trainer.SetupAdamWandB(m.b.h.XHandle(), decay1, decay2, int32(m.batchsize))
	if err != nil {
		return errors.New("(m *OutputModule) InitHiddenLayers(b *Builder, decay1,decay2 float32, batch int32)" + err.Error())
	}
	w.SetRates(rate, 0)
	bias.SetRates(rate, 0)

	err = m.op.LoadTrainer(m.b.h.Handler, m.batchsize, w, bias)
	if err != nil {
		return errors.New("(m *OutputModule) InitHiddenLayers(b *Builder, decay1,decay2 float32, batch int32)" + err.Error())
	}

	return nil
}

//InitWorkspace inits the workspace
func (m *OutputModule) InitWorkspace() (err error) {
	noerror := gocudnn.Status(0)
	var flag bool
	if m.op.cnn != nil {
		fwds, err := m.op.cnn.GetFwdAlgoPerfList(m.op.h.Handler, m.op.x.Tensor, m.op.y.Tensor, nil)
		for _, fwd := range fwds {
			if noerror == fwd.Status {

				m.op.cnn.SetFwdAlgoPerformance(fwd)
				if fwd.Memory > 0 {
					m.op.workspacefwd, err = nvidia.MallocGlobal(m.op.h.Handler, fwd.Memory)
					if err != nil {
						return err
					}
				}
				flag = true
				break
			}
		}
		if !flag {
			if performancedebugging {
				fmt.Println("fwds tensors")
				fmt.Println("X", m.op.x)
				fmt.Println("Y", m.op.y)
				fmt.Println("W", m.op.cnn)
				//for _, fwd := range fwds {
				//
				//fmt.Println(fwd)
				//
				//}
			}
			return errors.New("cnnInitForwardPerformanceFail")
		}
		flag = false
		bwds, err := m.op.cnn.GetBwdDataAlgoPerfList(m.op.h.Handler, m.op.x.Tensor, m.op.y.Tensor, nil)
		for _, bwd := range bwds {
			if noerror == bwd.Status {
				if performancedebugging {
					fmt.Println(bwd)
				}
				m.op.cnn.SetBwdDataAlgoPerformance(bwd)
				if bwd.Memory > 0 {
					m.op.workspacebwd, err = nvidia.MallocGlobal(m.op.h.Handler, bwd.Memory)
					if err != nil {
						return err
					}
				}
				flag = true
				break
			}
		}
		if !flag {
			if performancedebugging {
				for _, bwd := range bwds {

					fmt.Println(bwd)

				}
			}
			return errors.New("cnnInitBackwardDataPerformanceFail")
		}
		flag = false
		bwfs, err := m.op.cnn.GetBwdFiltAlgoPerfList(m.op.h.Handler, m.op.x.Tensor, m.op.y.Tensor, nil)
		for _, bwf := range bwfs {
			if noerror == bwf.Status {
				if performancedebugging {
					//	fmt.Println(bwf)
				}
				m.op.cnn.SetBwdFiltAlgoPerformance(bwf)
				if bwf.Memory > 0 {
					m.op.workspacebwf, err = nvidia.MallocGlobal(m.op.h.Handler, bwf.Memory)
					if err != nil {
						return err
					}
				}
				flag = true
				break
			}
		}
		if !flag {
			return errors.New("cnnInitBackwardFilterPerformanceFail")
		}
		flag = false
	} else if m.op.cnntranspose != nil {
		fwds, err := m.op.cnntranspose.GetFwdAlgoPerfList(m.op.h.Handler, m.op.x.Tensor, m.op.y.Tensor, nil)
		for _, fwd := range fwds {
			if noerror == fwd.Status {

				m.op.cnntranspose.SetFwdAlgoPerformance(fwd)
				if fwd.Memory > 0 {
					m.op.workspacefwd, err = nvidia.MallocGlobal(m.op.h.Handler, fwd.Memory)
					if err != nil {
						return err
					}
				}
				flag = true
				break
			}
		}
		if !flag {
			return errors.New("cnnInitForwardPerformanceFail")
		}
		flag = false
		bwds, err := m.op.cnntranspose.GetBwdDataAlgoPerfList(m.op.h.Handler, m.op.x.Tensor, m.op.y.Tensor, nil)
		for _, bwd := range bwds {
			if noerror == bwd.Status {

				m.op.cnntranspose.SetBwdDataAlgoPerformance(bwd)
				if bwd.Memory > 0 {
					m.op.workspacebwd, err = nvidia.MallocGlobal(m.op.h.Handler, bwd.Memory)
					if err != nil {
						return err
					}
				}
				flag = true
				break
			}
		}
		if !flag {
			return errors.New("cnnInitBackwardDataPerformanceFail")
		}
		flag = false
		bwfs, err := m.op.cnntranspose.GetBwdFiltAlgoPerfList(m.op.h.Handler, m.op.x.Tensor, m.op.y.Tensor, nil)
		for _, bwf := range bwfs {
			if noerror == bwf.Status {

				m.op.cnntranspose.SetBwdFiltAlgoPerformance(bwf)
				if bwf.Memory > 0 {
					m.op.workspacebwf, err = nvidia.MallocGlobal(m.op.h.Handler, bwf.Memory)
					if err != nil {
						return err
					}
				}
				flag = true
				break
			}
		}
		if !flag {
			return errors.New("cnnInitBackwardFilterPerformanceFail")
		}
		flag = false
	}
	return nil
}

//FindOutputDims satisifies module interface
func (m *OutputModule) FindOutputDims() ([]int32, error) {
	if m.op.x == nil {
		return nil, errors.New("m *OutputModule) FindOutputDims(): X tensor is not set")
	}
	if m.op.cnn != nil {
		return m.op.cnn.FindOutputDims(m.op.x.Tensor)
	}
	if m.op.cnntranspose != nil {
		return m.op.cnntranspose.FindOutputDims(m.op.x.Tensor)
	}
	return nil, errors.New("(m *OutputModule) FindOutputDims(): Major error both cnn and cnntranspose haven't been added")

}

//Update satisifies module interface
func (m *OutputModule) Update(epoch int) error {
	err := m.op.Update(epoch)
	if err != nil {
		return err
	}

	return nil
}

//Forward  satisfies module interface
func (m *OutputModule) Forward() error {

	return m.op.Forward()
}

//Backward  satisfies module interface
func (m *OutputModule) Backward() error {

	return m.op.Backward()
}

//Inference satisfies module interface
func (m *OutputModule) Inference() error {

	return m.op.Forward()
}

//GetTensorX returns set x tensor
func (m *OutputModule) GetTensorX() (x *Tensor) { return m.op.x }

//GetTensorDX returns set dx tensor
func (m *OutputModule) GetTensorDX() (dx *Tensor) { return m.op.dx }

//GetTensorY returns set y tensor
func (m *OutputModule) GetTensorY() (y *Tensor) {

	return m.op.y
}

//GetTensorDY returns set dy tensor
func (m *OutputModule) GetTensorDY() (dy *Tensor) {

	return m.op.dy
}

//SetTensorX sets x tensor
func (m *OutputModule) SetTensorX(x *Tensor) { m.op.x = x }

//SetTensorDX sets dx tensor
func (m *OutputModule) SetTensorDX(dx *Tensor) { m.op.dx = dx }

//SetTensorY sets y tensor
func (m *OutputModule) SetTensorY(y *Tensor) {
	m.op.y = y
}

//SetTensorDY sets dy tensor
func (m *OutputModule) SetTensorDY(dy *Tensor) {
	m.op.dy = dy
}
