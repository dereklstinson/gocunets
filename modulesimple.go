package gocunets

import (
	"github.com/dereklstinson/cutil"
)

//SimpleModule is a module that concats several layers together when doing the forward and backward passes
/*type SimpleModule struct {
	id           int64
	c            *Concat
	numofconvs   int
	layers       []*Layer
	activ        *Layer
	workspace    []cutil.Mem
	x, dx, y, dy *Tensor
	concaty      *Tensor
	concatdy     *Tensor
	batchsize    int
}
type SimpleModuleInfo struct {
	ID         int64
	Layers     []*cnn.Info
	Activation *activation.Info
}*/
type privatemodule struct {
	id           int64
	c            *Concat
	numofconvs   int
	layers       []*Layer
	activ        *Layer
	workspace    []cutil.Mem
	x, dx, y, dy *Tensor
	concaty      *Tensor
	concatdy     *Tensor
	batchsize    int
}

//CreateSimpleModule will create a simple module with each of the convolution layers being in parallel.
//The parallel convolution will have the same hw, but the channels for each can be changed.
//The number of convolutions depends on the length of the channel array.
//Each convolution will have pad = ((dim-1)/2) *dilation. This will make the output for each convolution equal in the spacial dims.
//The strides are stuck at 2.
//
//This considers a stride of 2 spacial dims (hw) need to be odd.
//This will preform a deconvolution with the formula for the output tensor:
//
//N= batch;
//
//C = [neurons[0]+ ... +neurons[i]];
//
//H,W and more (spacial dims) = input
//
/*
func CreateSimpleModule(bldr *Builder, batch, inputchannels int32, hw, neurons []int32, falpha, fbeta, balpha, bbeta float64) (m *SimpleModule, err error) {
	m = new(SimpleModule)
	m.numofconvs = len(neurons)
	m.layers = make([]*Layer, m.numofconvs)
	frmtflg := bldr.Frmt
	m.batchsize = int(batch)
	if !compatableDimsHW(hw) {
		return nil, errors.New("CreateDecompressionModule(...): hw []int32 elements need to be 1+2n where n = 0,1,2,3 ... aka odd")
	}
	for i := range m.layers {
		filterdims := make([]int32, len(hw)+2)
		filterdims[0] = neurons[i]
		pads := make([]int32, len(hw))
		strides := make([]int32, len(hw))
		dilations := make([]int32, len(hw))

		if bldr.Frmt == frmtflg.NCHW() {
			filterdims[1] = inputchannels
			for j := 0; j < len(hw); j++ {
				dim := hw[j]
				dilation := int32(i + 1)
				pad := findpad(findnfordims(dim), dilation)
				filterdims[j+2] = hw[j]
				dilations[j] = dilation
				pads[j] = pad
				strides[j] = 1
			}
		} else if bldr.Frmt == frmtflg.NHWC() {
			for j := 0; j < len(hw); j++ {
				filterdims[j+1] = hw[j]
				dim := hw[j]
				dilation := int32(i + 1)
				pad := findpad(findnfordims(dim), dilation)
				filterdims[j+1] = hw[j]
				filterdims[j+1] = hw[j]
				dilations[j] = dilation
				pads[j] = pad
				strides[j] = 1

			}
			filterdims[len(filterdims)-1] = inputchannels
		} else {
			return nil, errors.New(" CreateSimpleModule(bldr *Builder, nhw, channels []int32) : unnsupported format given from bldr")
		}
		w, dw, b, db, err := bldr.CreateConvolutionWeights(filterdims)
		if err != nil {
			return nil, err
		}

		m.layers[i], err = bldr.ConvolutionLayer(int64(i), 1, w, dw, b, db, pads, strides, dilations)
		if err != nil {
			return nil, err
		}
		m.layers[i].SetForwardScalars(falpha, 0)
		m.layers[i].SetBackwardScalars(balpha, bbeta)

	}
	m.c, err = CreateConcat(bldr.h)
	if err != nil {
		return nil, err
	}
	m.c.c.SetForwardAlpha(1)
	m.c.c.SetForwardBeta(fbeta)
	m.c.c.SetBackwardAlpha(1)
	m.c.c.SetBackwardBeta(0)
	m.activ, err = bldr.Activation(int64(len(m.layers)))
	return m, nil
}

//ID is the id
func (m *SimpleModule) ID() int64 {
	return m.id
}

//Name is the name string
func (m *SimpleModule) Name() string {
	return "SimpleModule"
}

//Forward does the forward operation
func (m *SimpleModule) Forward() (err error) {
	for i := range m.layers {
		err = m.layers[i].forwardprop()
		if err != nil {
			return err
		}
	}
	srcs := make([]*Tensor, len(m.layers))
	for i := range m.layers {
		srcs[i] = m.layers[i].y
	}
	err = m.c.Forward(srcs, m.y)
	if err != nil {
		return err
	}
	err = m.activ.forwardprop()
	if err != nil {
		return err
	}
	return nil
}

//Backward does the backward propagation
func (m *SimpleModule) Backward() (err error) {
	err = m.activ.backpropdata()
	if err != nil {
		return err
	}
	srcs := make([]*Tensor, len(m.layers))

	for i := range m.layers {
		srcs[i] = m.layers[i].dy
	}
	err = m.c.Backward(srcs, m.dy)
	if err != nil {
		return err
	}
	for i := range m.layers {
		err = m.layers[i].backpropfilterdata()
		if err != nil {
			return err
		}
	}

	if err != nil {
		return err
	}
	return nil
}

//Update updates the weights of the hidden convolution layer
func (m *SimpleModule) Update() (err error) {
	for _, l := range m.layers {
		err = l.updateWeights()
		if err != nil {
			return err
		}
	}
	return nil
}

//FindOutputDims returns the output dims of the module
func (m *SimpleModule) FindOutputDims() (dims []int32, err error) {
	if m.x == nil {
		return nil, errors.New(" (m *SimpleModule) FindOutputDims() : input tensor is nil")
	}

	preconcatdims := make([][]int32, len(m.layers))
	for i := range m.layers {
		preconcatdims[i], err = m.layers[i].GetOutputDims(m.x)
		if err != nil {
			return nil, err
		}
	}

	dims, err = m.c.c.GetOutputDimsfromInputDims(preconcatdims, m.x.Tensor.Format())
	return dims, err
}

//InitHiddenLayers will init the hidden layers. If
func (m *SimpleModule) InitHiddenLayers(b *Builder, decay1, decay2 float32, batch int32) (err error) {
	if m.x == nil || m.dy == nil || m.y == nil {
		return errors.New("(m *SimpleModule) InitHiddenLayers(): inputtensor x is nil || dy is nil || y is nil")
	}

	m.batchsize = int(batch)
	for i := range m.layers {

		outputdims, err := m.layers[i].GetOutputDims(m.x)
		if err != nil {
			return err
		}
		sharedYandDY, err := b.CreateTensor(outputdims)
		if err != nil {
			return err
		}
		m.layers[i].x, m.layers[i].dx = m.x, m.dx
		m.layers[i].y, m.layers[i].dy = sharedYandDY, sharedYandDY

		w, bias, err := trainer.SetupAdamWandB(b.h.XHandle(), decay1, decay2, batch)
		if err != nil {
			return errors.New("(m *SimpleModule) InitHiddenLayers(b *Builder, decay1,decay2 float32, batch int32)" + err.Error())
		}
		err = m.layers[i].LoadTrainer(b.h.Handler, int(batch), w, bias)
		if err != nil {
			return errors.New("(m *SimpleModule) InitHiddenLayers(b *Builder, decay1,decay2 float32, batch int32)" + err.Error())
		}

	}
	m.activ.dx, m.activ.dy = m.dy, m.dy
	m.activ.y, m.activ.x = m.y, m.x
	return nil
}

//InitWorkspace inits the hidden workspace
func (m *SimpleModule) InitWorkspace(b *Builder) (err error) {
	noerror := gocudnn.Status(0)
	for _, l := range m.layers {
		fwd, bwdd, bwdf, err := l.getcudnnperformance(b.h.Handler, l.x.Tensor, l.y.Tensor, nil)
		if err != nil {
			return err
		}
		var flag bool
		for i := range fwd {

			if noerror == fwd[i].Status {
				l.setcudnnperformancefwd(fwd[i])
				flag = true
				break
			}

		}
		if !flag {
			return errors.New("InitForwardPerformanceFail")
		}
		flag = false
		for i := range bwdd {

			if noerror == bwdd[i].Status {
				l.setcudnnperformancebwdd(bwdd[i])
				break
			}

		}
		if !flag {
			return errors.New("InitBackwardPerformanceDataFail")
		}
		flag = false
		for i := range bwdf {

			if noerror == bwdf[i].Status {

				l.setcudnnperformancebwdf(bwdf[i])

				break
			}

		}
		if !flag {
			return errors.New("InitBackwardPerformanceFilterFail")
		}
	}
	return nil
}

//GetTensorX returns set x tensor
func (m *SimpleModule) GetTensorX() (x *Tensor) { return m.x }

//GetTensorDX returns set dx tensor
func (m *SimpleModule) GetTensorDX() (dx *Tensor) { return m.dx }

//GetTensorY returns set y tensor
func (m *SimpleModule) GetTensorY() (y *Tensor) { return m.y }

//GetTensorDY returns set dy tensor
func (m *SimpleModule) GetTensorDY() (dy *Tensor) { return m.dy }

//SetTensorX sets x tensor
func (m *SimpleModule) SetTensorX(x *Tensor) { m.x = x }

//SetTensorDX sets dx tensor
func (m *SimpleModule) SetTensorDX(dx *Tensor) { m.dx = dx }

//SetTensorY sets y tensor
func (m *SimpleModule) SetTensorY(y *Tensor) { m.y = y }

//SetTensorDY sets dy tensor
func (m *SimpleModule) SetTensorDY(dy *Tensor) { m.dy = dy }
*/
