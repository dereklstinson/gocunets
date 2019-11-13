package gocunets

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/softmax"
	"github.com/dereklstinson/GoCuNets/trainer"
	"github.com/dereklstinson/GoCudnn/cudart"

	"github.com/dereklstinson/GoCuNets/layers/dropout"

	"errors"
	"fmt"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	act "github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/activation"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/batchnorm"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/cnntranspose"
	"github.com/dereklstinson/GoCuNets/layers/pooling"
	gocudnn "github.com/dereklstinson/GoCudnn"
	"math/rand"
	//	"strings"
	"time"
)

//IO is contains 2 tensors the x and dx.  Input IOs will contain only the X tensor.
type IO struct {
	*layers.IO
}

//CreateInputIO creates the input IO that only contains an x tensor
func (n *Network) CreateInputIO(dims []int32) (input *IO, err error) {
	x, err := layers.BuildNetworkInputIO(n.handle, n.frmt, n.dtype, dims)
	input.IO = x
	return input, err
}

//CreateIO creates an IO that holds both the x and dx tensor
func (n *Network) CreateIO(dims []int32) (output *IO, err error) {
	x, err := layers.BuildIO(n.handle, n.frmt, n.dtype, dims)
	output.IO = x
	return output, err
}

//CreateInferenceIO creates an inference IO
func (n *Network) CreateInferenceIO(dims []int32) (inference *IO, err error) {
	inference.IO, err = layers.BuildInferenceIO(n.handle, n.frmt, n.dtype, dims)

	return inference, err
}

//DataType struct wrapper for gocudnn.Datatype.  Look up methods in gocudnn.
type DataType struct {
	gocudnn.DataType
}

//TensorFormat struct wrapper for gocudnn.TensorFormat.  Look up methods in gocudnn.
type TensorFormat struct {
	gocudnn.TensorFormat
}

//ConvolutionMode struct wrapper for gocudnn.ConvolutionMode.  Look up methods in gocudnn.
type ConvolutionMode struct {
	gocudnn.ConvolutionMode
}

//NanProp struct wrapper for gocudnn.NanProp.  Look up methods in gocudnn.
type NanProp struct {
	gocudnn.NANProp
}

//BatchNormMode struct wrapper for gocudnn.BatchNormMode.  Look up methods in gocudnn.
type BatchNormMode struct {
	gocudnn.BatchNormMode
}

//BatchNormOps struct wrapper for gocudnn.BatchNormOps.  Look up methods in gocudnn.
type BatchNormOps struct {
	gocudnn.BatchNormOps
}

//PoolingMode struct wrapper for gocudnn.PoolingMode.  Look up methods in gocudnn.
type PoolingMode struct {
	gocudnn.PoolingMode
}

//ActivationMode struct wrapper for gocudnn.ActivationMode.  Look up methods in gocudnn.
type ActivationMode struct {
	act.Mode
}

//SoftmaxAlgo determins what algo to use for softmax
type SoftmaxAlgo struct {
	gocudnn.SoftMaxAlgorithm
}

//SoftmaxMode determins what mode to use for softmax
type SoftmaxMode struct {
	gocudnn.SoftMaxMode
}

//MathType is math type for tensor cores
type MathType struct {
	gocudnn.MathType
}

//Workspace is workspace used by the hidden layers
type Workspace struct {
	*nvidia.Malloced
}

//Trainer is a trainer.Trainer
type Trainer interface {
	trainer.Trainer
}

func trainerstooriginal(t []Trainer) (x []trainer.Trainer) {
	x = make([]trainer.Trainer, len(t))
	for i := range t {
		x[i] = t[i]
	}
	return x
}

/*
type ElementwiseOp struct {
	gocudnn.OpTensorOp
}
*/

//Flags is a struct that should only be used for passing flags.
var Flags struct {
	Format TensorFormat
	Dtype  DataType
	Nan    NanProp
	CMode  ConvolutionMode
	BNMode BatchNormMode
	BNOps  BatchNormOps
	PMode  PoolingMode
	AMode  ActivationMode
	SMMode SoftmaxMode
	SMAlgo SoftmaxAlgo
	MType  MathType
	//EleOp  ElementwiseOp
}
var pflags struct { //This is just in case someone was stupid enough to use Flags other than its intended purpose
	Format gocudnn.TensorFormat
	Dtype  gocudnn.DataType
	Nan    gocudnn.NANProp
	CMode  gocudnn.ConvolutionMode
	BNMode gocudnn.BatchNormMode
	BNOps  gocudnn.BatchNormOps
	PMode  gocudnn.PoolingMode
	AMode  act.ModeFlag
	SMmode gocudnn.SoftMaxMode
	SMAlgo gocudnn.SoftMaxAlgorithm
	//EleOp  gocudnn.OpTensorOp
}

//Handle handles the functions of the libraries used in gocunet
type Handle struct {
	*cudnn.Handler
}

//Stream is a stream for gpu instructions
type Stream struct {
	*cudart.Stream
}

//CreateHandle creates a handle for gocunets
func CreateHandle(d Device) (h *Handle) {
	h = new(Handle)
	h.Handler = cudnn.CreateHandler(d.Device)
	return h
}

//CreateStream creates a stream
func CreateStream() (s *Stream, err error) {
	s = new(Stream)
	s.Stream, err = cudart.CreateBlockingStream()
	return s, err
}

//Device is a gpu device
type Device struct {
	cudart.Device
}

//GetDeviceList gets a device from a list
func GetDeviceList() (devices []Device, err error) {
	n, err := cudart.GetDeviceCount()
	if err != nil {
		return nil, err
	}
	devices = make([]Device, n)
	for i := (int32)(0); i < n; i++ {
		devices[i].Device, err = cudart.CreateDevice(i)
		if err != nil {
			return nil, err
		}
	}
	return devices, nil
}

//CreateWorkSpace creates a workspace
func (n *Network) CreateWorkSpace(sib uint) (w *Workspace, err error) {
	w.Malloced, err = nvidia.MallocGlobal(n.handle, sib)
	return w, err
}

//CreateNetworkEX is a new way to create a network
//Use Flags global variable to pass flags into function
//
//example x:=CreateNetworkEX(h,Flags.Format.NHWC(), Flags.Dtype.Float32(),Flags.Cmode.CrossCorilation())
func CreateNetworkEX(handle *Handle, frmt TensorFormat, dtype DataType, cmode ConvolutionMode, mtype MathType) *Network {
	var x DataType
	y := x.Int8()
	fmt.Println(y)
	n := CreateNetwork()
	n.handle = handle.Handler
	n.cmode = cmode.ConvolutionMode
	n.frmt = frmt.TensorFormat
	n.dtype = dtype.DataType
	n.mathtype = mtype.MathType
	n.rngsource = rand.NewSource(time.Now().Unix())
	n.rng = rand.New(n.rngsource)
	n.nanprop = Flags.Nan.NotPropigate()

	return n
}

//InitializeEx uses the handler set in n
func (n *Network) InitializeEx(input, output *IO, wspace *Workspace) ([]ConvolutionPerformance, error) {
	return n.Initialize(n.handle, input.IO, output.IO, wspace.Malloced)
}

//LoadTrainersEx loads trainers
func (n *Network) LoadTrainersEx(trainers []Trainer) error {
	tw := trainerstooriginal(trainers)

	return n.LoadTrainers(n.handle, tw)
}

//ForwardPropEx does the forward prop for a prebuilt network
func (n *Network) ForwardPropEx(x, y *IO) error {
	return n.ForwardProp(n.handle, x.IO, y.IO)
}

//BackPropFilterDataEX does the backprop of the hidden layers
func (n *Network) BackPropFilterDataEX(x, y *IO) error {
	return n.BackPropFilterData(n.handle, x.IO, y.IO)
}

//ZeroHiddenInferenceIOsEX zeros out the hidden inference ios
func (n *Network) ZeroHiddenInferenceIOsEX() error {
	return n.ZeroHiddenInferenceIOs(n.handle)
}

//ZeroHiddenTrainingIOsEX zeros out the hidden training ios
func (n *Network) ZeroHiddenTrainingIOsEX() error {
	return n.ZeroHiddenTrainingIOs(n.handle)
}

//ZeroHiddenIOsEX will zero out the hidden ios.
// This is used for training the feedback loops for the scalars.
func (n *Network) ZeroHiddenIOsEX() error {
	return n.ZeroHiddenIOs(n.handle)
}

//UpdateWeightsEX updates the weights of a Network
func (n *Network) UpdateWeightsEX(batch int) error {
	return n.UpdateWeights(n.handle, batch)
}

//AppendSoftMax will append a softmax layer
func (n *Network) AppendSoftMax(sm SoftmaxMode, sa SoftmaxAlgo) (err error) {
	var layer *softmax.Layer
	switch sm.SoftMaxMode {
	case pflags.SMmode.Channel():
		switch sa.SoftMaxAlgorithm {
		case pflags.SMAlgo.Accurate():
			layer = softmax.StageAccuratePerChannel(nil)
		case pflags.SMAlgo.Fast():
			layer = softmax.StageFastPerChannel(nil)
		case pflags.SMAlgo.Log():
			layer = softmax.StageLogPerChannel(nil)
		default:
			return errors.New("Unsupported Mode,Algo")
		}
	case pflags.SMmode.Instance():
		switch sa.SoftMaxAlgorithm {
		case pflags.SMAlgo.Accurate():
			layer = softmax.StageAccuratePerInstance(nil)
		case pflags.SMAlgo.Fast():
			layer = softmax.StageFastPerInstance(nil)
		case pflags.SMAlgo.Log():
			layer = softmax.StageLogPerInstance(nil)
		default:
			return errors.New("Unsupported Mode,Algo")
		}

	}
	n.AddLayer(layer, nil)
	return nil
}

//AppendConvolution appends a convolution layer to the network
func (n *Network) AppendConvolution(filter, padding, stride, dilation []int32) (err error) {
	conv, err := cnn.Setup(n.handle, n.frmt, n.dtype, filter, n.cmode, padding, stride, dilation, n.rng.Uint64())
	conv.SetMathType(n.mathtype)
	n.AddLayer(conv, err)
	return err
}

//AppendTransposeConvolution appends a transpose convolution layer to the network
func (n *Network) AppendTransposeConvolution(filter, padding, stride, dilation []int32) (err error) {
	conv, err := cnntranspose.ReverseBuild(n.handle, n.frmt, n.dtype, filter, n.cmode, padding, stride, dilation, n.rng.Uint64())
	n.AddLayer(conv, err)
	return err
}

//AppendBatchNormalizaion appends a BatchNormalizaion layer to the network
func (n *Network) AppendBatchNormalizaion(BNMode gocudnn.BatchNormMode) (err error) {
	var bn *batchnorm.Layer
	switch BNMode {
	case pflags.BNMode.PerActivation():
		bn, err = batchnorm.PerActivationPreset(n.handle)
	case pflags.BNMode.Spatial():
		bn, err = batchnorm.SpatialPersistantPreset(n.handle, true)
	case pflags.BNMode.SpatialPersistent():
		bn, err = batchnorm.SpatialPreset(n.handle, true)
	default:
		err = errors.New("AppendBatchNormalizaion: unsupported mode")
	}
	n.AddLayer(bn, err)
	return err
}

//AppendActivation appends a Activation layer to the network
func (n *Network) AppendActivation(mode ActivationMode) (err error) {
	var act *activation.Layer

	switch mode.Mode {
	case pflags.AMode.Leaky():
		act, err = activation.Leaky(n.handle, n.dtype)
	case pflags.AMode.ClippedRelu():
		act, err = activation.ClippedRelu(n.handle, n.dtype)
	case pflags.AMode.Relu():
		act, err = activation.Relu(n.handle, n.dtype)
	case pflags.AMode.Elu():
		act, err = activation.Elu(n.handle, n.dtype)
	case pflags.AMode.Threshhold():
		act, err = activation.Threshhold(n.handle, n.dtype, -.2, -.001, -2, 2, 1, 3, true)
	case pflags.AMode.Sigmoid():
		act, err = activation.Sigmoid(n.handle, n.dtype)
	case pflags.AMode.Tanh():
		act, err = activation.Tanh(n.handle, n.dtype)
	case pflags.AMode.PRelu():
		act, err = activation.PRelu(n.handle, n.dtype, true)
	default:
		return errors.New("AppendActivation:  Not supported Activation Layer")
	}
	n.AddLayer(act, err)
	return err

}

//AppendPooling appends ap pooling layer to the network
func (n *Network) AppendPooling(mode PoolingMode, window, padding, stride []int32) (err error) {
	pool, err := pooling.SetupNoOutput(mode.PoolingMode, n.nanprop, window, padding, stride, true)
	n.AddLayer(pool, err)
	return err

}

//AppendReversePooling appends a reverse pooling layer to the network.
func (n *Network) AppendReversePooling(mode PoolingMode, window, padding, stride []int32) (err error) {
	pool, err := pooling.SetupNoOutputReverse(mode.PoolingMode, n.nanprop, window, padding, stride, true)
	n.AddLayer(pool, err)
	return err
}

//AppendDropout appends a Dropout layer to the network
func (n *Network) AppendDropout(drop float32) (err error) {
	do, err := dropout.Preset(n.handle, drop, n.rng.Uint64())
	n.AddLayer(do, err)
	return err
}

//OpAddForward performs the op into off the sources into dest.  dest elements will be set to zero before operation begins.
func (n *Network) OpAddForward(srcs []*IO, dest *IO) (err error) {
	err = dest.T().Memer().SetAll(0)
	if err != nil {
		return err
	}
	size := len(srcs)
	var iseven bool
	if size%2 == 0 {
		iseven = true
	} else {
		size--
	}
	for i := 0; i < size; i += 2 {
		n.handle.Sync()
		err = dest.T().OpAdd(n.handle, srcs[i].T(), dest.T(), 1, 1, 1)
		if err != nil {
			return err
		}
	}
	if !iseven {
		n.handle.Sync()
		err = dest.T().OpAdd(n.handle, srcs[size].T(), dest.T(), 1, 1, 0)
		if err != nil {
			return err
		}
	}
	return nil
}

//OpAddBackward doesn't perform an add operation.  It just backpropigates the errors back to the srcs Delta T.
func (n *Network) OpAddBackward(Dsrcs []*IO, Ddest *IO) (err error) {
	sib := Ddest.DeltaT().CurrentSizeT()
	err = n.handle.Sync()
	if err != nil {
		return err
	}
	for i := range Dsrcs {
		err = Dsrcs[i].DeltaT().LoadMem(n.handle, Ddest.DeltaT().Memer(), sib)
		if err != nil {
			return err
		}
	}

	return n.handle.Sync()
}
