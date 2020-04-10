package roman

import (
	"fmt"

	gocunets "github.com/dereklstinson/GoCuNets"
)

const dropoutpercent = float32(.2)

//RomanDecoder using regular method of increasing size of convolution...by just increasing the outer padding
func RomanDecoder(
	builder *gocunets.Builder,
	batchsize int32,
	outputchannel int32,
	hiddenoutputchannels []int32,
	learningrates, decay1, decay2 float32,
	x, dx *gocunets.Tensor) (mnet *gocunets.SimpleModuleNetwork) {
	mnet = gocunets.CreateSimpleModuleNetwork(2, builder)
	mods := make([]gocunets.Module, 7)

	var channeladder int32
	for i := range hiddenoutputchannels {
		channeladder += hiddenoutputchannels[i]
	}
	inputchannel := tensorchannelsize(x)
	if inputchannel < 1 {
		panic("input tensor channel is less than 1 or non supported tensor format ")
	}
	var err error
	mods[0], err = gocunets.CreateSingleStridedModule(0, builder, batchsize, inputchannel, hiddenoutputchannels, []int32{4, 4}, -3, 1, 0, false, true)
	if err != nil {
		panic(err)
	}
	mods[1], err = gocunets.CreateDecompressionModule(1, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 2, 1, 0)
	if err != nil {
		panic(err)
	}
	mods[2], err = gocunets.CreateDecompressionModule(2, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 2, 1, 0)
	if err != nil {
		panic(err)
	}
	mods[3], err = gocunets.CreateSingleStridedModule(3, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 1, 1, 0, false, true)
	if err != nil {
		panic(err)
	}
	mods[4], err = gocunets.CreateDecompressionModule(4, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 2, 1, 0)
	if err != nil {
		panic(err)
	}
	mods[5], err = gocunets.CreateDecompressionModule(5, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 2, 1, 0)
	if err != nil {
		panic(err)
	}
	mods[6], err = gocunets.CreateSingleStridedModule(6, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, -1, 1, 0, false, true)
	if err != nil {
		panic(err)
	}
	mnet.SetModules(mods)

	mnet.SetTensorX(x)
	mnet.SetTensorDX(dx)
	outputdims, err := mnet.FindOutputDims()
	if err != nil {
		panic(err)
	}
	//THis has to be NCHW
	fmt.Println("OutputDims", outputdims)

	outputfdims := []int32{outputchannel, channeladder, 3, 3}
	mnet.Output, err = gocunets.CreateOutputModule(7, builder, batchsize, outputfdims, []int32{1, 1}, []int32{1, 1}, []int32{1, 1}, 1, 0, 1, 0)
	if err != nil {
		panic(err)
	}
	err = mnet.SetSoftMaxClassifier()
	if err != nil {
		panic(err)
	}
	outputdims, err = mnet.FindOutputDims()
	if err != nil {
		panic(err)
	}
	fmt.Println("NewOutputDims", outputdims)
	ohy, err := builder.CreateTensor(outputdims)
	if err != nil {
		panic(err)
	}
	mnet.SetTensorY(ohy)
	ohdy, err := builder.CreateTensor(outputdims)
	if err != nil {
		panic(err)
	}
	mnet.SetTensorDY(ohdy)

	err = mnet.InitHiddenLayers(decay1, decay2)
	if err != nil {
		panic(err)
	}
	err = mnet.InitWorkspace()
	if err != nil {
		panic(err)
	}
	return mnet
}
func tensorchannelsize(x *gocunets.Tensor) int32 {

	flg := x.Format()
	switch x.Format() {
	case flg.NCHW():
		xdims := x.Dims()
		return xdims[1]
	case flg.NHWC():
		xdims := x.Dims()
		return xdims[len(xdims)-1]
	default:
		return -1
	}

}

//ArabicEncoder encodes the arabic
func ArabicEncoder(
	builder *gocunets.Builder,
	batchsize int32,
	outputchannel int32,
	hiddenoutputchannels []int32,
	learningrates, decay1, decay2 float32,
	x *gocunets.Tensor) (mnet *gocunets.SimpleModuleNetwork) {
	var channeladder int32
	for i := range hiddenoutputchannels {
		channeladder += hiddenoutputchannels[i]
	}

	mnet = gocunets.CreateSimpleModuleNetwork(0, builder)
	mods := make([]gocunets.Module, 7)
	inputchannel := tensorchannelsize(x)
	if inputchannel < 1 {
		panic("input tensor channel is less than 1 or non supported tensor format ")
	}
	var err error
	mods[0], err = gocunets.CreateSingleStridedModule(0, builder, batchsize, inputchannel, hiddenoutputchannels, []int32{4, 4}, -1, 1, 0, false, false)
	if err != nil {
		panic(err)
	}
	mods[1], err = gocunets.CreateCompressionModule(1, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 2, 1, 0)
	if err != nil {
		panic(err)
	}
	mods[2], err = gocunets.CreateCompressionModule(2, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 2, 1, 0)
	if err != nil {
		panic(err)
	}
	mods[3], err = gocunets.CreateSingleStridedModule(3, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 1, 1, 0, false, false)
	if err != nil {
		panic(err)
	}
	mods[4], err = gocunets.CreateCompressionModule(4, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 2, 1, 0)
	if err != nil {
		panic(err)
	}
	mods[5], err = gocunets.CreateCompressionModule(5, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 2, 1, 0)
	if err != nil {
		panic(err)
	}
	mods[6], err = gocunets.CreateSingleStridedModule(6, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, -3, 1, 0, false, false)
	if err != nil {
		panic(err)
	}

	mnet.SetModules(mods)

	mnet.SetTensorX(x)
	outputdims, err := mnet.FindOutputDims()
	if err != nil {
		panic(err)
	}
	//THis has to be NCHW
	fmt.Println("OutputDims", outputdims)

	outputfdims := []int32{outputchannel, channeladder, 3, 3}
	mnet.Output, err = gocunets.CreateOutputModule(7, builder, batchsize, outputfdims, []int32{1, 1}, []int32{1, 1}, []int32{1, 1}, 1, 0, 1, 0)
	if err != nil {
		panic(err)
	}

	outputdims, err = mnet.FindOutputDims()
	if err != nil {
		panic(err)
	}
	fmt.Println("NewOutputDims", outputdims)
	ohy, err := builder.CreateTensor(outputdims)
	if err != nil {
		panic(err)
	}
	mnet.SetTensorY(ohy)
	ohdy, err := builder.CreateTensor(outputdims)
	if err != nil {
		panic(err)
	}
	mnet.SetTensorDY(ohdy)

	err = mnet.InitHiddenLayers(decay1, decay2)
	if err != nil {
		panic(err)
	}
	err = mnet.InitWorkspace()
	if err != nil {
		panic(err)
	}
	return mnet
}

// output = 1+ (input +2*pad - (((filterdims-1)*dilation)+1))/stride
// ((output -1)*stride) - input - 2*pad +1 = ((filterdims-1)*dilation)
// if output == input && stride ==1
// output-1-input == -1 -> -1 -2*pad +1 = -2(pad)
//

// output = 1+ (input +2*pad - (((filterdims-1)*dilation)+1))/stride
// (output-1)*stride =input+2*pad - ((filterdims-1)*dilation+1)
// filter = 3
// (output-1)*stride = input + 2*(pad - dilation) -1
// pad = dilation
// output = (input -1)/stride + 1

// if variable filterdims it gets complicated
//
// (output-1)*stride = input + 2*pad - ((filterdims-1)*dilation+1)
//   				 = input + 2*pad - (((filterdims-1)*dilation) + 1) 	// filter dims = 1+2n
// 					 = input + 2*pad - (((2*n)*dilation) +1)
// 					 = input + 2*(pad - n*dilation) -1                  // if pad = (n*dilation) dilation > 0
//					 = input -1
//	output = ((input -1)/stride) +1	if filterdims = 3 + 2n && pad = (n+1)*dilation	// This makes the output dependent on the stride.
//

//ArabicDecoder using regular method of increasing size of convolution...by just increasing the outer padding
func ArabicDecoder(builder *gocunets.Builder,
	batchsize int32,
	outputchannel int32,
	hiddenoutputchannels []int32,
	learningrates, decay1, decay2 float32,
	x, dx *gocunets.Tensor) (mnet *gocunets.SimpleModuleNetwork) {

	var channeladder int32
	for i := range hiddenoutputchannels {
		channeladder += hiddenoutputchannels[i]
	}
	mnet = gocunets.CreateSimpleModuleNetwork(1, builder)
	mods := make([]gocunets.Module, 7)
	inputchannel := tensorchannelsize(x)
	if inputchannel < 1 {
		panic("input tensor channel is less than 1 or non supported tensor format ")
	}
	var err error
	mods[0], err = gocunets.CreateSingleStridedModule(0, builder, batchsize, inputchannel, hiddenoutputchannels, []int32{4, 4}, -3, 1, 0, false, true)
	if err != nil {
		panic(err)
	}
	mods[1], err = gocunets.CreateDecompressionModule(1, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 2, 1, 0)
	if err != nil {
		panic(err)
	}
	mods[2], err = gocunets.CreateDecompressionModule(2, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 2, 1, 0)
	if err != nil {
		panic(err)
	}
	mods[3], err = gocunets.CreateSingleStridedModule(3, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 1, 1, 0, false, true)
	if err != nil {
		panic(err)
	}
	mods[4], err = gocunets.CreateDecompressionModule(4, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 2, 1, 0)
	if err != nil {
		panic(err)
	}
	mods[5], err = gocunets.CreateDecompressionModule(5, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, 2, 1, 0)
	if err != nil {
		panic(err)
	}
	mods[6], err = gocunets.CreateSingleStridedModule(6, builder, batchsize, channeladder, hiddenoutputchannels, []int32{4, 4}, -1, 1, 0, false, true)
	if err != nil {
		panic(err)
	}

	mnet.SetModules(mods)
	mnet.SetTensorX(x)
	mnet.SetTensorDX(dx)
	outputdims, err := mnet.FindOutputDims()
	if err != nil {
		panic(err)
	}
	//THis has to be NCHW
	fmt.Println("OutputDims", outputdims)

	outputfdims := []int32{outputchannel, channeladder, 3, 3}
	mnet.Output, err = gocunets.CreateOutputModule(7, builder, batchsize, outputfdims, []int32{1, 1}, []int32{1, 1}, []int32{1, 1}, 1, 0, 1, 0)
	if err != nil {
		panic(err)
	}
	err = mnet.SetSoftMaxClassifier()
	if err != nil {
		panic(err)
	}
	outputdims, err = mnet.FindOutputDims()
	if err != nil {
		panic(err)
	}
	fmt.Println("NewOutputDims", outputdims)
	ohy, err := builder.CreateTensor(outputdims)
	if err != nil {
		panic(err)
	}
	mnet.SetTensorY(ohy)
	ohdy, err := builder.CreateTensor(outputdims)
	if err != nil {
		panic(err)
	}
	mnet.SetTensorDY(ohdy)

	err = mnet.InitHiddenLayers(decay1, decay2)
	if err != nil {
		panic(err)
	}
	err = mnet.InitWorkspace()
	if err != nil {
		panic(err)
	}
	return mnet

}
