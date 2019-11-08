package gocunets

import (
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
}

//CreateNetworkEX is a new way to create a network
//Use Flags global variable to pass flags into function
//
//example x:=CreateNetworkEX(h,Flags.Format.NHWC(), Flags.Dtype.Float32(),Flags.Cmode.CrossCorilation())
func CreateNetworkEX(handle *cudnn.Handler, frmt TensorFormat, dtype DataType, cmode ConvolutionMode) *Network {
	var x DataType
	y := x.Int8()
	fmt.Println(y)
	n := CreateNetwork()
	n.handle = handle
	n.cmode = cmode.ConvolutionMode
	n.frmt = frmt.TensorFormat
	n.dtype = dtype.DataType
	n.rngsource = rand.NewSource(time.Now().Unix())
	n.rng = rand.New(n.rngsource)
	n.nanprop = Flags.Nan.NotPropigate()
	return n
}

//AppendConvolution appends a convolution layer to the network
func (n *Network) AppendConvolution(filter, padding, stride, dilation []int32) (err error) {
	conv, err := cnn.Setup(n.handle, n.frmt, n.dtype, filter, n.cmode, padding, stride, dilation, n.rng.Uint64())
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
