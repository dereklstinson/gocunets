package gocunets

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/cutil"
	"github.com/dereklstinson/gocudnn/curand"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia"
	"github.com/dereklstinson/gocunets/layers"
	"github.com/dereklstinson/gocunets/layers/activation"
	"github.com/dereklstinson/gocunets/layers/batchnorm"
	"github.com/dereklstinson/gocunets/layers/cnn"
	"github.com/dereklstinson/gocunets/layers/cnntranspose"
	"github.com/dereklstinson/gocunets/layers/dropout"
	"github.com/dereklstinson/gocunets/layers/pooling"
)

//Builder will create layers with the flags set within the struct
type Builder struct {
	h         *Handle
	gpurng    *curand.Generator
	Frmt      TensorFormat
	Dtype     DataType
	Cmode     ConvolutionMode
	Mtype     MathType
	Pmode     PoolingMode
	AMode     ActivationMode
	BNMode    BatchNormMode
	Nan       NanProp
	curngtype curand.RngType
}

var bprflags struct {
	Frmt   TensorFormat
	Dtype  DataType
	Cmode  ConvolutionMode
	Mtype  MathType
	Pmode  PoolingMode
	Nan    NanProp
	BNMode BatchNormMode
}

//CreateBuilder creates a Builder.  Flags can be set by flags' methods inside of Builder.
//Default Flags are set at:
//
//	Frmt.NCHW()
//
//	Mtype.Default()
//
//	Nan.NotPropigate()
//
//	Cmode.CrossCorrelation()
//
//	Dtype.Float()
//
//	Pmode.AverageCountExcludePadding()
//
//  BNMode.Spatial()
//
//	AMode.Leaky()
func CreateBuilder(h *Handle) (b *Builder) {
	b = new(Builder)
	b.h = h
	b.Frmt.NCHW()
	b.Mtype.Default()
	b.Cmode.CrossCorrelation()
	b.Nan.NotPropigate()
	b.Dtype.Float()
	b.AMode.Leaky()
	b.Pmode.AverageCountExcludePadding()
	b.BNMode.Spatial()
	b.curngtype.PseudoDefault()
	b.gpurng = curand.CreateGeneratorEx(b.h.Handler.Worker, b.curngtype)
	return b
}

//GetHandle returns the handle
func (l *Builder) GetHandle() *Handle {
	return l.h
}

//AllocateMemory allocates memory
func (l *Builder) AllocateMemory(sib uint) (cutil.Pointer, error) {
	return nvidia.MallocGlobal(l.h.Handler.Worker, sib)
}

//CreateTensor creates a tensor initialed with all zeros.
func (l *Builder) CreateTensor(dims []int32) (t *Tensor, err error) {
	t = new(Tensor)
	t.Tensor, err = layers.CreateTensor(l.h.Handler, l.Frmt.TensorFormat, l.Dtype.DataType, dims)
	if err != nil {
		err = fmt.Errorf(" (l *Builder) CreateTensor, Err: %v, input dims: %v", err, dims)
	}
	return t, err

}

//CreateRandomTensor creates a random gaussian tensor.
func (l *Builder) CreateRandomTensor(dims []int32, mean, std float32, seed uint64) (t *Tensor, err error) {
	t = new(Tensor)
	t.Tensor, err = layers.BuildRandomTensor(l.h.Handler, l.Frmt.TensorFormat, l.Dtype.DataType, dims, mean, std)
	return t, err

}

//CreateBiasTensor is a helper function that will create the bias tensor considering the weight dims.
func (l *Builder) CreateBiasTensor(weightdims []int32, deconvolution bool) (b *Tensor, err error) {
	var biasdims = make([]int32, len(weightdims))
	for i := range biasdims {
		biasdims[i] = 1
	}

	//biasdims[0] = dims[0] //this is number of batches
	fmtflg := l.Frmt
	switch l.Frmt {
	case fmtflg.NCHW():
		if deconvolution {
			biasdims[1] = weightdims[1]
		} else {
			biasdims[1] = weightdims[0]
		}

	case fmtflg.NHWC():
		if deconvolution {
			biasdims[len(weightdims)-1] = weightdims[len(weightdims)-1]
		} else {
			biasdims[len(weightdims)-1] = weightdims[0]
		}

	default:
		return nil, errors.New("unsupported tensor format in builder")
	}
	b, err = l.CreateTensor(biasdims)
	if err != nil {
		return nil, err
	}
	return b, nil

}

//CreateDeconvolutionWeights creates the weights and delta weights of a deconvolution layer
func (l *Builder) CreateDeconvolutionWeights(dims []int32) (w, dw, b, db *Tensor, err error) {
	w, err = l.CreateTensor(dims)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	dw, err = l.CreateTensor(dims)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	b, err = l.CreateBiasTensor(w.Dims(), true)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	db, err = l.CreateBiasTensor(dw.Dims(), true)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return w, dw, b, db, nil
}

//CreateConvolutionWeights creates the weights and delta weights of a convolution layer
func (l *Builder) CreateConvolutionWeights(dims []int32) (
	w, dw, b, db *Tensor, err error) {
	w, err = l.CreateTensor(dims)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	dw, err = l.CreateTensor(dims)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	b, err = l.CreateBiasTensor(w.Dims(), false)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	db, err = l.CreateBiasTensor(dw.Dims(), false)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return w, dw, b, db, nil
}

/*
//ConnectLayers Creates the output of layer1 and connects it as the input to layer2.
func (l *Builder) ConnectLayers(layer1, layer2 *Layer) error {
	var err error
	var t *Tensor
	var dt *Tensor
	if layer1.x == nil {
		return errors.New("(l *Builder) ConnectLayers(layer1, layer2 *Layer)error :  layer1 doesn't have an input set")
	}
	dims, err := layer1.GetOutputDims((layer1.y))
	if err != nil {
		return err
	}
	t, err = l.CreateTensor(dims)
	if err != nil {
		return err
	}

	dt, err = l.CreateTensor(dims)
	if err != nil {
		return err
	}
	//	t.to=layer2
	//	t.from=layer1
	//	dt.to=layer2
	//	dt.from=layer1
	layer1.SetTensorY(t)
	layer1.SetTensorDY(dt)
	layer2.SetTensorX(t)
	layer2.SetTensorDX(dt)

	return nil
}
*/

//PoolingLayer creates a pooling layer with flags set in Builder
func (l *Builder) PoolingLayer(id int64, window, padding, stride []int32) (p *Layer, err error) {

	player, err := pooling.SetupNoOutput(l.Pmode.PoolingMode, l.Nan.NANProp, window, padding, stride)
	if err != nil {
		return nil, err
	}
	p, err = createlayer(id, l.h, player)

	return
}

//BatchNorm is the batch norm layer
func (l *Builder) BatchNorm(id int64) (batch *Layer, err error) {
	var blayer *batchnorm.Layer
	bprflg := bprflags
	switch l.BNMode {
	case bprflg.BNMode.PerActivation():
		blayer, err = batchnorm.PerActivationPreset(l.h.Handler)
	case bprflg.BNMode.Spatial():
		blayer, err = batchnorm.SpatialPreset(l.h.Handler)
	case bprflg.BNMode.SpatialPersistent():
		blayer, err = batchnorm.SpatialPersistantPreset(l.h.Handler)
	}
	if err != nil {
		return nil, err
	}
	batch, err = createlayer(id, l.h, blayer)

	return batch, err
}

//ConvolutionLayer creates a convolution layer
func (l *Builder) ConvolutionLayer(id int64, groupcount int32, w, dw, b, db *Tensor, pad, stride, dilation []int32) (conv *Layer, err error) {
	clayer, err := cnn.SetupBasic(
		l.h.Handler,
		l.Frmt.TensorFormat,
		l.Dtype.DataType,
		l.Mtype.MathType,
		groupcount,
		w.Tensor, dw.Tensor, b.Tensor, db.Tensor,
		l.Cmode.ConvolutionMode,
		pad,
		stride,
		dilation)
	if err != nil {
		return nil, err
	}
	conv, err = createlayer(id, l.h, clayer)
	if err != nil {
		return nil, err
	}
	return conv, nil
}

//Dropout creates an Dropout layer
func (l *Builder) Dropout(id int64, dropoutpercent float32, seed uint64) (d *Layer, err error) {
	dlayer, err := dropout.Preset(l.h.Handler, dropoutpercent, seed)
	if err != nil {
		return nil, err
	}
	d, err = createlayer(id, l.h, dlayer)

	return d, err
}

//Activation creates an activation layer
func (l *Builder) Activation(id int64) (a *Layer, err error) {
	var act *activation.Layer
	aflg := l.AMode
	switch l.AMode {
	case aflg.Leaky():
		act, err = activation.Leaky(l.h.Handler, l.Dtype.DataType)
	case aflg.ClippedRelu():
		act, err = activation.ClippedRelu(l.h.Handler, l.Dtype.DataType)
	case aflg.Relu():
		act, err = activation.Relu(l.h.Handler, l.Dtype.DataType)
	case aflg.Elu():
		act, err = activation.Elu(l.h.Handler, l.Dtype.DataType)
	case aflg.Threshhold():
		act, err = activation.Threshhold(l.h.Handler, l.Dtype.DataType, -.2, -.001, -2, 2, 1, 3, true)
	case aflg.Sigmoid():
		act, err = activation.Sigmoid(l.h.Handler, l.Dtype.DataType)
	case aflg.Tanh():
		act, err = activation.Tanh(l.h.Handler, l.Dtype.DataType)
	case aflg.PRelu():
		act, err = activation.PRelu(l.h.Handler, l.Dtype.DataType, true)
	default:
		return nil, errors.New("AppendActivation:  Not supported Activation Layer")
	}
	if err != nil {
		return nil, err
	}
	a, err = createlayer(id, l.h, act)

	return a, err
}

//DeConvolutionLayer creates a reverse convolution layer
func (l *Builder) DeConvolutionLayer(id int64, groupcount int32, w, dw, b, db *Tensor, pad, stride, dilation []int32) (rconv *Layer, err error) {
	clayer, err := cnntranspose.SetupBasic(l.h.Handler,
		l.Frmt.TensorFormat,
		l.Dtype.DataType,
		l.Mtype.MathType,
		groupcount,
		w.Tensor,
		dw.Tensor,
		b.Tensor,
		db.Tensor,
		l.Cmode.ConvolutionMode,
		pad, stride, dilation)
	if err != nil {
		return nil, err
	}
	rconv, err = createlayer(id, l.h, clayer)
	return rconv, err
}
