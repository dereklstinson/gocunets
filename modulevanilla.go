package gocunets

import (
	"errors"
	"fmt"

	gocudnn "github.com/dereklstinson/gocudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia"
	"github.com/dereklstinson/gocunets/trainer"
)

//VanillaModule has a convolution and an activation
type VanillaModule struct {
	id        int64
	b         *Builder
	conv      *Layer
	act       *Layer
	batchsize int
}

//ID satisfies module interface
func (m *VanillaModule) ID() int64 {
	return m.id
}

//func(m *VanillaModule)GetWeightandDweightTensors(){
//m.conv.cnn.DeltaWeights().TogglePrintValueForStringer()
//
//}

//CreateVanillaModule creates an output module
func CreateVanillaModule(id int64, bldr *Builder, batch int32, fdims, pad, stride, dilation []int32, balpha, bbeta, falpha, fbeta float64) (m *VanillaModule, err error) {
	m = new(VanillaModule)
	m.b = bldr
	m.id = id

	m.batchsize = int(batch)

	w, dw, b, db, err := m.b.CreateConvolutionWeights(fdims)
	if err != nil {
		return nil, err
	}
	m.conv, err = m.b.ConvolutionLayer(0, 1, w, dw, b, db, pad, stride, dilation)
	if err != nil {
		return nil, err
	}
	m.conv.SetBackwardScalars(balpha, bbeta)
	m.conv.SetOtherScalars(1, 0)
	m.conv.SetForwardScalars(1, 0)
	m.act, err = m.b.Activation(1)
	if err != nil {
		return nil, err
	}
	m.act.SetBackwardScalars(1, 0)
	m.act.SetForwardScalars(falpha, fbeta)
	return m, nil
}

//InitHiddenLayers will init the hidden operation
func (m *VanillaModule) InitHiddenLayers(rate, decay1, decay2 float32) (err error) {

	if m.conv.cnn != nil {
		err := m.conv.cnn.MakeRandom(m.conv.h.Handler, m.conv.x.Dims())
		if err != nil {
			return err
		}

	} else if m.conv.cnntranspose != nil {
		err := m.conv.cnntranspose.MakeRandom(m.conv.h.Handler, m.conv.x.Dims())
		if err != nil {
			return err
		}

	} else {
		return errors.New("(m *VanillaModule)InitHiddenLayers. CreateModule needs to be ran first")
	}
	odims, err := m.conv.GetOutputDims(m.conv.x)
	if err != nil {
		return err
	}
	m.conv.y, err = m.b.CreateTensor(odims)
	if err != nil {
		return err
	}
	m.conv.dy, err = m.b.CreateTensor(odims)
	if err != nil {
		return err
	}

	m.act.x = m.conv.y
	m.act.dx = m.conv.dy

	err = m.b.h.Sync()
	if err != nil {
		return err
	}
	w, bias, err := trainer.SetupAdamWandB(m.b.h.XHandle(), decay1, decay2, int32(m.batchsize))
	if err != nil {
		return errors.New("(m *VanillaModule) InitHiddenLayers(b *Builder, decay1,decay2 float32, batch int32)" + err.Error())
	}
	w.SetRates(rate, 0)
	bias.SetRates(rate, 0)
	err = m.conv.LoadTrainer(m.b.h.Handler, m.batchsize, w, bias)
	if err != nil {
		return errors.New("(m *VanillaModule) InitHiddenLayers(b *Builder, decay1,decay2 float32, batch int32)" + err.Error())
	}

	return nil
}

//InitWorkspace inits the workspace
func (m *VanillaModule) InitWorkspace() (err error) {
	noerror := gocudnn.Status(0)
	var flag bool
	if m.conv.cnn != nil {
		fwds, err := m.conv.cnn.GetFwdAlgoPerfList(m.conv.h.Handler, m.conv.x.Tensor, m.conv.y.Tensor, nil)
		for _, fwd := range fwds {
			if noerror == fwd.Status {

				m.conv.cnn.SetFwdAlgoPerformance(fwd)
				if fwd.Memory > 0 {
					m.conv.workspacefwd, err = nvidia.MallocGlobal(m.conv.h.Handler, fwd.Memory)
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
				fmt.Println("X", m.conv.x)
				fmt.Println("Y", m.conv.y)
				fmt.Println("W", m.conv.cnn)
				//for _, fwd := range fwds {
				//
				//fmt.Println(fwd)
				//
				//}
			}
			return errors.New("cnnInitForwardPerformanceFail")
		}
		flag = false
		bwds, err := m.conv.cnn.GetBwdDataAlgoPerfList(m.conv.h.Handler, m.conv.x.Tensor, m.conv.y.Tensor, nil)
		for _, bwd := range bwds {
			if noerror == bwd.Status {
				if performancedebugging {
					fmt.Println(bwd)
				}
				m.conv.cnn.SetBwdDataAlgoPerformance(bwd)
				if bwd.Memory > 0 {
					m.conv.workspacebwd, err = nvidia.MallocGlobal(m.conv.h.Handler, bwd.Memory)
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
		bwfs, err := m.conv.cnn.GetBwdFiltAlgoPerfList(m.conv.h.Handler, m.conv.x.Tensor, m.conv.y.Tensor, nil)
		for _, bwf := range bwfs {
			if noerror == bwf.Status {
				if performancedebugging {
					//	fmt.Println(bwf)
				}
				m.conv.cnn.SetBwdFiltAlgoPerformance(bwf)
				if bwf.Memory > 0 {
					m.conv.workspacebwf, err = nvidia.MallocGlobal(m.conv.h.Handler, bwf.Memory)
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
	} else if m.conv.cnntranspose != nil {
		fwds, err := m.conv.cnntranspose.GetFwdAlgoPerfList(m.conv.h.Handler, m.conv.x.Tensor, m.conv.y.Tensor, nil)
		for _, fwd := range fwds {
			if noerror == fwd.Status {

				m.conv.cnntranspose.SetFwdAlgoPerformance(fwd)
				if fwd.Memory > 0 {
					m.conv.workspacefwd, err = nvidia.MallocGlobal(m.conv.h.Handler, fwd.Memory)
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
		bwds, err := m.conv.cnntranspose.GetBwdDataAlgoPerfList(m.conv.h.Handler, m.conv.x.Tensor, m.conv.y.Tensor, nil)
		for _, bwd := range bwds {
			if noerror == bwd.Status {

				m.conv.cnntranspose.SetBwdDataAlgoPerformance(bwd)
				if bwd.Memory > 0 {
					m.conv.workspacebwd, err = nvidia.MallocGlobal(m.conv.h.Handler, bwd.Memory)
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
		bwfs, err := m.conv.cnntranspose.GetBwdFiltAlgoPerfList(m.conv.h.Handler, m.conv.x.Tensor, m.conv.y.Tensor, nil)
		for _, bwf := range bwfs {
			if noerror == bwf.Status {

				m.conv.cnntranspose.SetBwdFiltAlgoPerformance(bwf)
				if bwf.Memory > 0 {
					m.conv.workspacebwf, err = nvidia.MallocGlobal(m.conv.h.Handler, bwf.Memory)
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

//FindOutputDims satisfies module interface
func (m *VanillaModule) FindOutputDims() ([]int32, error) {
	if m.conv.x == nil {
		return nil, errors.New("m *VanillaModule) FindOutputDims(): X tensor is not set")
	}
	if m.conv.cnn != nil {
		return m.conv.cnn.FindOutputDims(m.conv.x.Tensor)
	}
	if m.conv.cnntranspose != nil {
		return m.conv.cnntranspose.FindOutputDims(m.conv.x.Tensor)
	}
	return nil, errors.New("(m *VanillaModule) FindOutputDims(): Major error both cnn and cnntranspose haven't been added")

}

//Update satisfies module interface
func (m *VanillaModule) Update(epoch int) error {
	err := m.conv.Update(epoch)
	if err != nil {
		return err
	}

	return m.act.Update(epoch)

}

//Forward  satisfies module interface
//
func (m *VanillaModule) Forward() error {

	err := m.conv.Forward()
	if err != nil {
		return err
	}
	return m.act.Forward()
}

//Backward  satisfies module interface
func (m *VanillaModule) Backward() error {

	err := m.act.Backward()
	if err != nil {
		return err
	}
	return m.conv.Backward()

}

//Inference satisfies module interface
func (m *VanillaModule) Inference() error {

	err := m.conv.Forward()
	if err != nil {
		return err
	}
	return m.act.Forward()
}

//GetTensorX returns set x tensor
func (m *VanillaModule) GetTensorX() (x *Tensor) { return m.conv.x }

//GetTensorDX returns set dx tensor
func (m *VanillaModule) GetTensorDX() (dx *Tensor) { return m.conv.dx }

//GetTensorY returns set y tensor
func (m *VanillaModule) GetTensorY() (y *Tensor) {

	return m.act.y

}

//GetTensorDY returns set dy tensor
func (m *VanillaModule) GetTensorDY() (dy *Tensor) {
	return m.act.dy
}

//SetTensorX sets x tensor
func (m *VanillaModule) SetTensorX(x *Tensor) { m.conv.x = x }

//SetTensorDX sets dx tensor
func (m *VanillaModule) SetTensorDX(dx *Tensor) { m.conv.dx = dx }

//SetTensorY sets y tensor
func (m *VanillaModule) SetTensorY(y *Tensor) {
	m.act.y = y
}

//SetTensorDY sets dy tensor
func (m *VanillaModule) SetTensorDY(dy *Tensor) {
	m.act.dy = dy

}
