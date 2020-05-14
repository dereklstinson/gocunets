package gocunets

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia"

	gocudnn "github.com/dereklstinson/gocudnn"
	//	"github.com/dereklstinson/cutil"
)

type module struct {
	id              int64
	c               *Concat
	b               *Builder
	layers          []*Layer
	activ           *Layer
	x, dx, y, dy    *Tensor
	batchsize       int
	deconvolutional bool
}
type initialization struct {
	dims                         []int32
	strides                      []int32
	batch                        int32
	neurons                      []int32
	falpha, fbeta, balpha, bbeta float64
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
func createModule(id int64, bldr *Builder,
	batch, inputchannels int32, outputchannels []int32,
	spacialdims []int32,
	paddingoffset int32,
	falpha, fbeta float64,
	strides, deconvolution bool) (m *module, err error) {
	m = new(module)
	m.b = bldr
	m.id = id
	m.batchsize = int(batch)
	m.deconvolutional = deconvolution

	m.layers = make([]*Layer, len(outputchannels))

	var stride = int32(1)
	if strides {
		stride = 2
	}
	//divback := float64(len(m.layers))
	if deconvolution {
		for i := range m.layers {
			filterdims, pads, strides, dilations, err := deconvolutionparameterdims(inputchannels,
				outputchannels[i],
				stride,
				spacialdims,
				paddingoffset,
				bldr.Frmt,
				i)
			//	fmt.Println("filterdims,pads,strides,dilations", filterdims, pads, strides, dilations)
			if err != nil {
				return nil, err
			}
			w, dw, b, db, err := bldr.CreateDeconvolutionWeights(filterdims)
			if err != nil {
				return nil, err
			}

			m.layers[i], err = bldr.DeConvolutionLayer(int64(i), 1, w, dw, b, db, pads, strides, dilations)
			if err != nil {
				return nil, err
			}

			m.layers[i].SetForwardScalars(1, 0)
			m.layers[i].SetOtherScalars(1, 0)
			m.layers[i].SetBackwardScalars(1, 1)

		}

	} else {
		for i := range m.layers {
			//fmt.Println("outputchannels, stride spacialdims, paddingoffset, fmt, index", outputchannels[i], stride, spacialdims, paddingoffset, bldr.Frmt, i)
			filterdims, pads, strides, dilations, err := convolutionparameterdims(inputchannels,
				outputchannels[i],
				stride,
				spacialdims,
				paddingoffset,
				bldr.Frmt,
				i)
			if err != nil {
				return nil, err
			}
			w, dw, b, db, err := bldr.CreateConvolutionWeights(filterdims)
			if err != nil {
				return nil, err
			}

			m.layers[i], err = bldr.ConvolutionLayer(int64(i), 1, w, dw, b, db, pads, strides, dilations)
			if err != nil {
				return nil, err
			}
			m.layers[i].SetForwardScalars(1, 0)
			m.layers[i].SetOtherScalars(1, 0)
			m.layers[i].SetBackwardScalars(1, 1)

		}
	}

	m.c, err = CreateConcat(bldr.h)
	if err != nil {
		return nil, err
	}
	m.c.c.SetForwardAlpha(1)
	m.c.c.SetForwardBeta(0)
	m.c.c.SetBackwardAlpha(1)
	m.c.c.SetBackwardBeta(0)
	m.activ, err = bldr.Activation(int64(len(m.layers)))
	if err != nil {
		return nil, err
	}
	m.activ.activation.SetForwardScalars(falpha, fbeta)
	m.activ.activation.SetBackwardScalars(1, 0)
	return m, nil
}
func convolutionparameterdims(inputchannels, outputchannel, stride int32,
	spacialdims []int32, paddingoffset int32, frmt TensorFormat, index int) (fdims, pads, strides, dils []int32, err error) {
	fdims = make([]int32, len(spacialdims)+2)
	fdims[0] = outputchannel
	pads = make([]int32, len(spacialdims))
	strides = make([]int32, len(spacialdims))
	dils = make([]int32, len(spacialdims))
	flg := frmt
	switch frmt {
	case flg.NCHW():
		fdims[1] = inputchannels //output channel size for deconv is the neuron channels
		for i := 0; i < len(spacialdims); i++ {
			dim := spacialdims[i]

			dilation, pad, err := recommendedpaddilation(dim, (int32)(index), stride, paddingoffset)
			if err != nil {
				return nil, nil, nil, nil, err
			}
			fdims[i+2] = dim
			dils[i] = dilation
			pads[i] = pad
			strides[i] = stride
		}

	case flg.NHWC():
		for i := 0; i < len(spacialdims); i++ {
			fdims[i+1] = spacialdims[i]
			dim := spacialdims[i]

			dilation, pad, err := recommendedpaddilation(dim, (int32)(index), stride, paddingoffset)
			if err != nil {
				return nil, nil, nil, nil, err
			}
			fdims[i+1] = dim
			dils[i] = dilation
			pads[i] = pad
			strides[i] = stride

		}

		fdims[len(fdims)-1] = inputchannels
	default:
		return nil, nil, nil, nil, errors.New("Unsupported Format")
	}
	return fdims, pads, strides, dils, nil
}
func deconvolutionparameterdims(inputchannels,
	outputchannel,
	stride int32,
	spacialdims []int32,
	paddingoffset int32,
	frmt TensorFormat,
	index int) (fdims, pads, strides, dils []int32, err error) {
	fdims = make([]int32, len(spacialdims)+2)
	//convolution filter of NCHW (OuputChannels,Inputchannels h,w) could actually be thought of
	//reverse convolution would be (InputChannels,OutputChannels, h,w)
	fdims[0] = inputchannels
	pads = make([]int32, len(spacialdims))
	strides = make([]int32, len(spacialdims))
	dils = make([]int32, len(spacialdims))
	flg := frmt
	switch frmt {
	case flg.NCHW():

		fdims[1] = outputchannel //output channel size for deconv is the neuron channels
		for i := 0; i < len(spacialdims); i++ {
			dim := spacialdims[i]
			dilation, pad, err := recommendedpaddilation(dim, (int32)(index), stride, paddingoffset)
			if err != nil {
				return nil, nil, nil, nil, err
			}

			fdims[i+2] = dim
			dils[i] = dilation
			pads[i] = pad
			strides[i] = stride
		}

	case flg.NHWC():
		for i := 0; i < len(spacialdims); i++ {
			fdims[i+1] = spacialdims[i]
			dim := spacialdims[i]
			dilation, pad, err := recommendedpaddilation(dim, (int32)(index), stride, paddingoffset)
			if err != nil {
				return nil, nil, nil, nil, err
			}
			fdims[i+2] = dim
			dils[i] = dilation
			pads[i] = pad
			strides[i] = stride

		}
		fdims[len(fdims)-1] = outputchannel
	default:
		return nil, nil, nil, nil, errors.New("Unsupported Format")
	}
	return fdims, pads, strides, dils, nil
}

//ID is the id
func (m *module) ID() int64 {
	return m.id
}

//Forward does the forward operation
func (m *module) Forward() (err error) {
	for i := range m.layers {
		err = m.layers[i].forwardprop()
		if err != nil {
			if moduleforwarddebugging {
				fmt.Println("error on forward index:", i)
				fmt.Println(m.layers[i])
			}

			return err

		}

	}
	err = m.c.Forward()
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
func (m *module) Backward() (err error) {
	if m.dx != nil {

		err = m.dx.SetValues(m.b.h.Handler, 0)
		if err != nil {
			return err
		}
	}

	err = m.activ.backpropdata()
	if err != nil {
		return err
	}
	if moduleactivationdebugging {
		m.activ.dx.TogglePrintValueForStringer()
		fmt.Println("ActivationDX", m.activ.dx)
		m.activ.dx.TogglePrintValueForStringer()
	}

	err = m.c.Backward()
	if err != nil {
		return err
	}
	if moduleconcatdebugging {
		//	m.concatdy.TogglePrintValueForStringer()
		//	fmt.Println("ConcatDy", m.concatdy)
		//	m.concatdy.TogglePrintValueForStringer()
		for _, src := range m.c.deltasrcs {
			src.TogglePrintValueForStringer()
			fmt.Println("Concat Src: ", src)
			src.TogglePrintValueForStringer()
		}
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

//OutputDims returns the output dims of the module
func (m *module) OutputDims() (dims []int32, err error) {
	if m.x == nil {
		return nil, errors.New(" (m *SimpleModule) OutputDims() : input tensor is nil")
	}

	preconcatdims := make([][]int32, len(m.layers))
	for i := range m.layers {
		preconcatdims[i], err = m.layers[i].OutputDims()
		if err != nil {
			return nil, err
		}
		if moduleconcatdebugging {
			fmt.Println("Preconcatdims", preconcatdims[i])
		}
	}

	dims, err = m.c.c.GetOutputDimsfromInputDims(preconcatdims, m.x.Tensor.Format())
	return dims, err
}

//InitHiddenLayers will init the hidden layers. If
func (m *module) InitHiddenLayers() (err error) {
	if m.x == nil || m.dy == nil || m.y == nil {
		return errors.New("(m *module) InitHiddenLayers(): inputtensor x is nil || dy is nil || y is nil")
	}

	for i := range m.layers {

		outputdims, err := m.layers[i].OutputDims()
		if err != nil {
			return err
		}
		//	m.layers[i].x, m.layers[i].dx = m.x, m.dx
		m.layers[i].y, err = m.b.CreateTensor(outputdims)
		if err != nil {
			return err
		}
		m.layers[i].dy, err = m.b.CreateTensor(outputdims)
		if err != nil {
			return err
		}

		if m.layers[i].cnn != nil {
			err = m.layers[i].cnn.MakeRandom(m.layers[i].h.Handler, m.layers[i].x.Dims())

		} else if m.layers[i].cnntranspose != nil {
			err = m.layers[i].cnntranspose.MakeRandom(m.layers[i].h.Handler, m.layers[i].x.Dims())

		}

		if err != nil {
			return err
		}

	}
	srcs := make([]*Tensor, len(m.layers))
	deltasrcs := make([]*Tensor, len(m.layers))
	for i := range m.layers {

		srcs[i] = m.layers[i].y
		deltasrcs[i] = m.layers[i].dy
	}
	m.c.SetInputSrcs(srcs)
	m.c.SetInputDeltaSrcs(deltasrcs)
	outputdims, err := m.c.OutputDims()
	if err != nil {
		return err
	}
	concaty, err := m.b.CreateTensor(outputdims)
	if err != nil {
		return err
	}
	concatdy, err := m.b.CreateTensor(outputdims)
	if err != nil {
		return err
	}
	m.c.SetDest(concaty)
	m.c.SetDeltaDest(concatdy)
	m.activ.dy, m.activ.dx = m.dy, concatdy
	m.activ.y, m.activ.x = m.y, concaty
	return nil
}

var performancedebugging bool

//PerformanceDebugging - if function called it raises flag to print performance for inner layers
func PerformanceDebugging() {
	performancedebugging = true
}

//InitWorkspace inits the hidden workspace
func (m *module) InitWorkspace() (err error) {
	noerror := gocudnn.Status(0)
	var flag bool
	for _, l := range m.layers {
		if l.cnn != nil {
			fwds, err := l.cnn.GetFwdAlgoPerfList(l.h.Handler, l.x.Tensor, l.y.Tensor, nil)
			for _, fwd := range fwds {
				if noerror == fwd.Status {
					if performancedebugging {
						fmt.Println(fwd)
					}
					l.cnn.SetFwdAlgoPerformance(fwd)
					if fwd.Memory > 0 {
						l.workspacefwd, err = nvidia.MallocGlobal(l.h.Handler, fwd.Memory)
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
			bwds, err := l.cnn.GetBwdDataAlgoPerfList(l.h.Handler, l.x.Tensor, l.y.Tensor, nil)
			for _, bwd := range bwds {
				if noerror == bwd.Status {
					if performancedebugging {
						fmt.Println(bwd)
					}
					l.cnn.SetBwdDataAlgoPerformance(bwd)
					if bwd.Memory > 0 {
						l.workspacebwd, err = nvidia.MallocGlobal(l.h.Handler, bwd.Memory)
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
			bwfs, err := l.cnn.GetBwdFiltAlgoPerfList(l.h.Handler, l.x.Tensor, l.y.Tensor, nil)
			for _, bwf := range bwfs {
				if noerror == bwf.Status {
					if performancedebugging {
						fmt.Println(bwf)
					}
					l.cnn.SetBwdFiltAlgoPerformance(bwf)
					if bwf.Memory > 0 {
						l.workspacebwf, err = nvidia.MallocGlobal(l.h.Handler, bwf.Memory)
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
		}

		if l.cnntranspose != nil {
			fwds, err := l.cnntranspose.GetFwdAlgoPerfList(l.h.Handler, l.x.Tensor, l.y.Tensor, nil)
			for _, fwd := range fwds {
				if noerror == fwd.Status {
					if performancedebugging {
						fmt.Println(fwd)
					}

					l.cnntranspose.SetFwdAlgoPerformance(fwd)
					if fwd.Memory > 0 {
						l.workspacefwd, err = nvidia.MallocGlobal(l.h.Handler, fwd.Memory)
						if err != nil {
							return err
						}
					}
					flag = true
					break
				}
			}
			if !flag {
				return errors.New("cnntransposeInitForwardPerformanceFail")
			}
			bwds, err := l.cnntranspose.GetBwdDataAlgoPerfList(l.h.Handler, l.x.Tensor, l.y.Tensor, nil)
			for _, bwd := range bwds {
				if noerror == bwd.Status {
					if performancedebugging {
						fmt.Println(bwd)
					}

					l.cnntranspose.SetBwdDataAlgoPerformance(bwd)
					if bwd.Memory > 0 {
						l.workspacebwd, err = nvidia.MallocGlobal(l.h.Handler, bwd.Memory)
						if err != nil {
							return err
						}
					}
					flag = true
					break
				}
			}
			if !flag {
				return errors.New("cnntransposeInitBackwardDataPerformanceFail")
			}
			bwfs, err := l.cnntranspose.GetBwdFiltAlgoPerfList(l.h.Handler, l.x.Tensor, l.y.Tensor, nil)
			for _, bwf := range bwfs {

				if noerror == bwf.Status {
					if performancedebugging {
						fmt.Println(bwf)
					}
					l.cnntranspose.SetBwdFiltAlgoPerformance(bwf)
					if bwf.Memory > 0 {
						l.workspacebwf, err = nvidia.MallocGlobal(l.h.Handler, bwf.Memory)
						if err != nil {
							return err
						}
					}
					flag = true
					break
				}
			}
			if !flag {
				return errors.New("cnntransposeInitBackwardFilterPerformanceFail")
			}
		}
	}

	return nil
}

//Inference does the inference forward operation
func (m *module) Inference() (err error) {
	return m.Forward()
}

//GetTensorX returns set x tensor
func (m *module) GetTensorX() (x *Tensor) { return m.x }

//GetTensorDX returns set dx tensor
func (m *module) GetTensorDX() (dx *Tensor) { return m.dx }

//GetTensorY returns set y tensor
func (m *module) GetTensorY() (y *Tensor) { return m.y }

//GetTensorDY returns set dy tensor
func (m *module) GetTensorDY() (dy *Tensor) { return m.dy }

//SetTensorX sets x tensor
func (m *module) SetTensorX(x *Tensor) {
	m.x = x
	for i := range m.layers {
		m.layers[i].SetTensorX(x)
	}

}
func (m *module) GetWeights() []*Tensor {
	t := make([]*Tensor, 0)
	for _, layer := range m.layers {
		t = append(t, layer.GetWeights()...)
	}
	if m.activ.GetWeights() != nil {
		t = append(t, m.activ.GetWeights()...)
	}
	return t
}

//SetTensorDX sets dx tensor
func (m *module) SetTensorDX(dx *Tensor) {
	m.dx = dx
	for i := range m.layers {
		m.layers[i].SetTensorDX(dx)
	}
}

//SetTensorY sets y tensor
func (m *module) SetTensorY(y *Tensor) {
	m.y = y
	if m.activ != nil {
		m.activ.y = y
	}
}

//SetTensorDY sets dy tensor
func (m *module) SetTensorDY(dy *Tensor) {
	m.dy = dy
	if m.activ != nil {
		m.activ.dy = dy
	}
}
