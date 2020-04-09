package gocunets

//import (
//	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
//	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
//	"github.com/dereklstinson/GoCuNets/layers"
//	"github.com/dereklstinson/GoCuNets/trainer"
//	"github.com/dereklstinson/GoCudnn/cudart"
//	"github.com/dereklstinson/GoCudnn/gocu"
//	//	"strings"
//)

/*
//CreateTensor creates an IO that holds both the x and dx tensor
func (n *Network) CreateTensor(dims []int32) (output *Tensor, err error) {
	output = new(Tensor)
	output.Tensor, err = layers.CreateTensor(n.handle.Handler, n.frmt, n.dtype, dims)
	return output, err
}
*/
//Workspace is workspace used by the hidden layers

/*
type ElementwiseOp struct {
	gocudnn.OpTensorOp
}
*/

////CreateWorkSpace creates a workspace
//func (n *Network) CreateWorkSpace(sib uint) (w *Workspace, err error) {
//	w.Malloced, err = nvidia.MallocGlobal(n.handle, sib)
//	return w, err
//}

/*
//SetWSpaceSizeEx sets the fastest algorithm for the convolution based on the limit of the workspace.
func (n *Network) SetWSpaceSizeEx(wspacefwd, wspacebwdd, wspacebwdf uint, perfs []ConvolutionPerformance) {
	n.SetWSpaceSize(n.handle.Handler, wspacefwd, wspacebwdd, wspacebwdf, perfs)
}
*/

//CreateNetworkEX is a new way to create a network
//Use Flags global variable to pass flags into function
//
//example x:=CreateNetworkEX(h,Flags.Format.NHWC(), Flags.Dtype.Float32(),Flags.Cmode.CrossCorilation())
//func CreateNetworkEX(handle *Handle, frmt TensorFormat, dtype DataType, cmode ConvolutionMode, mtype MathType) *Network {
//	var x DataType
//	y := x.Int8()
//	fmt.Println(y)
//
//	n := CreateNetwork(handle)
//	n.handle = handle
//	n.cmode = cmode.ConvolutionMode
//	n.frmt = frmt.TensorFormat
//	n.dtype = dtype.DataType
//	n.mathtype = mtype.MathType
//	n.rngsource = rand.NewSource(time.Now().Unix())
//	n.rng = rand.New(n.rngsource)
//	n.nanprop = Flags.Nan.NANProp.NotPropigate()
//
//	return n
//}

/*
//InitializeEx uses the handler set in n
func (n *Network) InitializeEx(wspace *Workspace) ([]ConvolutionPerformance, error) {
	if wspace == nil {
		return n.Initialize(n.handle.Handler, nil)
	}
	return n.Initialize(n.handle.Handler, wspace.Malloced)
}
*/

////ForwardPropEx does the forward prop for a prebuilt network
//func (n *Network) ForwardPropEx() error {
//	return n.ForwardProp()
//}
//
////BackPropFilterDataEX does the backprop of the hidden layers
//func (n *Network) BackPropFilterDataEX() error {
//	return n.BackPropFilterData()
//}
//
////BackPropDataEx does the backprop on data only.  Please run all back prop datas first
//func (n *Network) BackPropDataEx() error {
//	return n.BackPropData()
//}
//
////BackPropFilterEx does the backprop filter only.  Please run all back prop datas first
//func (n *Network) BackPropFilterEx() error {
//	return n.BackPropFilter()
//}
//
////ZeroHiddenIOsEX will zero out the hidden ios.
//// This is used for training the feedback loops for the scalars.
//func (n *Network) ZeroHiddenIOsEX() error {
//	return n.ZeroHiddenIOs()
//}
//
////UpdateWeightsEX updates the weights of a Network
//func (n *Network) UpdateWeightsEX(epoch int) error {
//	return n.UpdateWeights(epoch)
//}
//
////AppendSoftMax will append a softmax layer
//func (n *Network) AppendSoftMax(sm SoftmaxMode, sa SoftmaxAlgo) (err error) {
//	var layer *softmax.Layer
//	switch sm.SoftMaxMode {
//	case pflags.SMmode.Channel():
//		switch sa.SoftMaxAlgorithm {
//		case pflags.SMAlgo.Accurate():
//			layer = softmax.StageAccuratePerChannel(nil)
//		case pflags.SMAlgo.Fast():
//			layer = softmax.StageFastPerChannel(nil)
//		case pflags.SMAlgo.Log():
//			layer = softmax.StageLogPerChannel(nil)
//		default:
//			return errors.New("Unsupported Mode,Algo")
//		}
//	case pflags.SMmode.Instance():
//		switch sa.SoftMaxAlgorithm {
//		case pflags.SMAlgo.Accurate():
//			layer = softmax.StageAccuratePerInstance(nil)
//		case pflags.SMAlgo.Fast():
//			layer = softmax.StageFastPerInstance(nil)
//		case pflags.SMAlgo.Log():
//			layer = softmax.StageLogPerInstance(nil)
//		default:
//			return errors.New("Unsupported Mode,Algo")
//		}
//
//	}
//	l, err := createlayer(n.idcounter, n.handle, layer)
//	if err != nil {
//		return err
//	}
//	n.idcounter++
//	n.AddLayer(l)
//	return nil
//}

////AppendConvolution appends a convolution layer to the network
//func (n *Network) AppendConvolution(groupcount int32, filter, padding, stride, dilation []int32) (err error) {
//	conv, err := cnn.Setup(n.handle.Handler, n.frmt, n.dtype, n.mathtype, groupcount, filter, n.cmode, padding, stride, dilation, n.rng.Uint64())
//	//conv.SetMathType(n.mathtype)
//	l, err := createlayer(n.idcounter, n.handle, conv)
//	if err != nil {
//		return err
//	}
//	n.idcounter++
//	n.AddLayer(l)
//	return nil
//}

////AppendTransposeConvolution appends a transpose convolution layer to the network
//func (n *Network) AppendTransposeConvolution(groupcount int32, filter, padding, stride, dilation []int32) (err error) {
//	conv, err := cnntranspose.Setup(n.handle.Handler, n.frmt, n.dtype, n.mathtype, groupcount, filter, n.cmode, padding, stride, dilation, n.rng.Uint64())
//	l, err := createlayer(n.idcounter, n.handle, conv)
//	if err != nil {
//		return err
//	}
//	n.idcounter++
//	n.AddLayer(l)
//	return nil
//}
//
////AppendBatchNormalizaion appends a BatchNormalizaion layer to the network
//func (n *Network) AppendBatchNormalizaion(BNMode gocudnn.BatchNormMode) (err error) {
//	var bn *batchnorm.Layer
//	switch BNMode {
//	case pflags.BNMode.PerActivation():
//		bn, err = batchnorm.PerActivationPreset(n.handle.Handler)
//	case pflags.BNMode.Spatial():
//		bn, err = batchnorm.SpatialPersistantPreset(n.handle.Handler)
//	case pflags.BNMode.SpatialPersistent():
//		bn, err = batchnorm.SpatialPreset(n.handle.Handler)
//	default:
//		err = errors.New("AppendBatchNormalizaion: unsupported mode")
//	}
//	l, err := createlayer(n.idcounter, n.handle, bn)
//	if err != nil {
//		return err
//	}
//	n.idcounter++
//	n.AddLayer(l)
//	return nil
//}

////AppendActivation appends a Activation layer to the network
//func (n *Network) AppendActivation(mode ActivationMode) (err error) {
//	var act *activation.Layer
//
//	switch mode.Mode {
//	case pflags.AMode.Leaky():
//		act, err = activation.Leaky(n.handle.Handler, n.dtype)
//	case pflags.AMode.ClippedRelu():
//		act, err = activation.ClippedRelu(n.handle.Handler, n.dtype)
//	case pflags.AMode.Relu():
//		act, err = activation.Relu(n.handle.Handler, n.dtype)
//	case pflags.AMode.Elu():
//		act, err = activation.Elu(n.handle.Handler, n.dtype)
//	case pflags.AMode.Threshhold():
//		act, err = activation.Threshhold(n.handle.Handler, n.dtype, -.2, -.001, -2, 2, 1, 3, true)
//	case pflags.AMode.Sigmoid():
//		act, err = activation.Sigmoid(n.handle.Handler, n.dtype)
//	case pflags.AMode.Tanh():
//		act, err = activation.Tanh(n.handle.Handler, n.dtype)
//	case pflags.AMode.PRelu():
//		act, err = activation.PRelu(n.handle.Handler, n.dtype, true)
//	default:
//		return errors.New("AppendActivation:  Not supported Activation Layer")
//	}
//	l, err := createlayer(n.idcounter, n.handle, act)
//	if err != nil {
//		return err
//	}
//	n.idcounter++
//	n.AddLayer(l)
//	return nil
//
//}
//
////AppendPooling appends ap pooling layer to the network
//func (n *Network) AppendPooling(mode PoolingMode, window, padding, stride []int32) (err error) {
//	pool, err := pooling.SetupNoOutput(mode.PoolingMode, n.nanprop, window, padding, stride)
//	l, err := createlayer(n.idcounter, n.handle, pool)
//	if err != nil {
//		return err
//	}
//	n.idcounter++
//	n.AddLayer(l)
//	return nil
//}
//
////AppendReversePooling appends a reverse pooling layer to the network.
//func (n *Network) AppendReversePooling(mode PoolingMode, window, padding, stride []int32) (err error) {
//	pool, err := pooling.SetupNoOutputReverse(mode.PoolingMode, n.nanprop, window, padding, stride)
//	l, err := createlayer(n.idcounter, n.handle, pool)
//	if err != nil {
//		return err
//	}
//	n.idcounter++
//	n.AddLayer(l)
//	return nil
//}

////AppendDropout appends a Dropout layer to the network
//func (n *Network) AppendDropout(drop float32) (err error) {
//	do, err := dropout.Preset(n.handle.Handler, drop, n.rng.Uint64())
//	l, err := createlayer(n.idcounter, n.handle, do)
//	if err != nil {
//		return err
//	}
//	n.idcounter++
//	n.AddLayer(l)
//	return nil
//}
//
////OpAddForward performs the op into off the sources into dest.  dest elements will be set to zero before operation begins.
//func (n *Network) OpAddForward(srcs []*Tensor, dest *Tensor) (err error) {
//	err = dest.SetAll(0)
//	if err != nil {
//		return err
//	}
//	size := len(srcs)
//	var iseven bool
//	if size%2 == 0 {
//		iseven = true
//	} else {
//		size--
//	}
//	for i := 0; i < size; i += 2 {
//		n.handle.Sync()
//		err = dest.OpAdd(n.handle.Handler, srcs[i].Volume, dest.Volume, 1, 1, 1)
//		if err != nil {
//			return err
//		}
//	}
//	if !iseven {
//		n.handle.Sync()
//		err = dest.OpAdd(n.handle.Handler, srcs[size].Volume, dest.Volume, 1, 1, 0)
//		if err != nil {
//			return err
//		}
//	}
//	return nil
//}
//
////OpAddBackward doesn't perform an add operation.  It just backpropigates the errors back to the srcs Delta T.
//func (n *Network) OpAddBackward(Dsrcs []*Tensor, Ddest *Tensor) (err error) {
//	sib := Ddest.SIB()
//	err = n.handle.Sync()
//	if err != nil {
//		return err
//	}
//	for i := range Dsrcs {
//		err = Dsrcs[i].LoadMem(n.handle.Handler, Ddest, sib)
//		if err != nil {
//			return err
//		}
//	}
//
//	return n.handle.Sync()
//}
