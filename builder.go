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
	//	rng       *rand.Rand
	//	src       rand.Source
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
	//	b.src = rand.NewSource(seed)
	//	b.rng = rand.New(b.src)
	b.Frmt.NCHW()
	b.Mtype.Default()
	b.Cmode.CrossCorrelation()
	b.Nan.NotPropigate()
	b.Dtype.Float()
	b.AMode.Leaky()
	b.Pmode.AverageCountExcludePadding()
	b.BNMode.Spatial()
	b.curngtype.PseudoDefault()
	b.gpurng = curand.CreateGeneratorEx(b.h.w, b.curngtype)
	return b
}

//GetHandle returns the handle
func (l *Builder) GetHandle() *Handle {
	return l.h
}

//AllocateMemory allocates memory
func (l *Builder) AllocateMemory(sib uint) (cutil.Pointer, error) {
	return nvidia.MallocGlobal(l.h.w, sib)
}

//CreateTensor creates a tensor
func (l *Builder) CreateTensor(dims []int32) (t *Tensor, err error) {
	//	err = l.h.w.Work(func() error {
	t = new(Tensor)

	t.Tensor, err = layers.CreateTensor(l.h.Handler, l.Frmt.TensorFormat, l.Dtype.DataType, dims)
	if err != nil {
		err = fmt.Errorf(" (l *Builder) CreateTensor, Err: %v, input dims: %v", err, dims)
	}
	return t, err

}

//CreateRandomTensor creates a random tensor
func (l *Builder) CreateRandomTensor(dims []int32, mean, std float32, seed uint64) (t *Tensor, err error) {
	//	err = l.h.w.Work(func() error {
	//	var err1 error
	t = new(Tensor)
	t.Tensor, err = layers.BuildRandomTensor(l.h.Handler, l.Frmt.TensorFormat, l.Dtype.DataType, dims, mean, std)
	//return nil, err1
	//	})
	return t, err

}

//FindBiasTensor finds the bias tensor according to the dims
func (l *Builder) FindBiasTensor(dims []int32) (b *Tensor, err error) {
	//	err = l.h.w.Work(func() error {
	//var err1 error
	var biasdims = make([]int32, len(dims))
	for i := range biasdims {
		biasdims[i] = 1
	}
	biasdims[0] = dims[0] //this is number of batches
	switch l.Frmt {
	case bprflags.Frmt.NCHW():
		biasdims[1] = dims[1]

	case bprflags.Frmt.NHWC():
		biasdims[len(dims)-1] = dims[len(dims)-1]
	default:
		return nil, errors.New("unsupported tensor format in builder")
	}
	b, err = l.CreateTensor(biasdims)
	//	return nil, err1
	//	})
	return b, err

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
	flg := l.Frmt
	bdims := make([]int32, len(dims))
	for i := range bdims {
		bdims[i] = 1
	}
	switch l.Frmt {
	case flg.NCHW():
		bdims[1] = dims[1]

	case flg.NHWC():
		bdims[len(bdims)-1] = dims[len(dims)-1]
	default:
		return nil, nil, nil, nil, errors.New("(l *Builder) CreateDeconvolutionWeights: Unsupported Format")
	}

	b, err = l.CreateTensor(bdims)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	db, err = l.CreateTensor(bdims)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return w, dw, b, db, nil
}

//CreateConvolutionWeights creates the weights and delta weights of a convolution layer
func (l *Builder) CreateConvolutionWeights(dims []int32) (w, dw, b, db *Tensor, err error) {

	w, err = l.CreateTensor(dims)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	dw, err = l.CreateTensor(dims)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	flg := l.Frmt
	bdims := make([]int32, len(dims))
	for i := range bdims {
		bdims[i] = 1
	}
	switch l.Frmt {
	case flg.NCHW():
		bdims[1] = dims[0]

	case flg.NHWC():
		bdims[len(bdims)-1] = dims[0]
	default:
		return nil, nil, nil, nil, errors.New("(l *Builder) CreateConvolutionWeights: Unsupported Format")
	}

	b, err = l.CreateTensor(bdims)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	db, err = l.CreateTensor(bdims)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return w, dw, b, db, nil
}

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
	layer1.SetOutputs(t, dt)
	layer2.SetInputs(t, dt)

	return nil
}

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

	switch l.BNMode {
	case bprflags.BNMode.PerActivation():
		blayer, err = batchnorm.PerActivationPreset(l.h.Handler)
	case bprflags.BNMode.Spatial():
		blayer, err = batchnorm.SpatialPreset(l.h.Handler)
	case bprflags.BNMode.SpatialPersistent():
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
	//err = l.h.w.Work(func() error {
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
	//	return nil, nil
	//	})
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

//ReverseConvolutionLayer creates a reverse convolution layer
func (l *Builder) ReverseConvolutionLayer(id int64, groupcount int32, w, dw, b, db *Tensor, pad, stride, dilation []int32) (rconv *Layer, err error) {
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
