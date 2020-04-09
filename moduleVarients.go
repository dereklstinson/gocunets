package gocunets

//DecompressionModule is a module that concats several layers together when doing the forward and backward passes
type DecompressionModule struct {
	*module
}

//CreateDecompressionModule will create a simple module with each of the deconvolution layers being in parallel.
//Deconvolution output channels is determined by the size of the neuron channels. The number of neurons needs to equal the input channels.
//
//The parallel convolution will have the same hw, but the neuron channels for each can be changed.
//The number of convolutions depends on the length of the channel array.
//Each convolution will have pad = ((dim-1)/2) *dilation. This will make the output for each convolution equal in the spacial dims.
//The strides are stuck at 2.
//
//This considers a stride of 2 spacial dims (hw) need to be odd.
//This will preform a deconvolution with the formula for the output tensor:
//
//N= batch;
//
//C = [neuronchannels[0]+ ... +neuronchannels[i]];
//
//H,W and more (spacial dims) = 2*input - 1
//
//the dx tensor is zeroed before back propagation
// If multiple modules share the same input. Each module will need its own tensor as its own dx output. Those tensors need to be summed
//into the output dy tensor of module it got its x input tensor from
func CreateDecompressionModule(id int64, bldr *Builder, batch, inputchannel int32, outputperparallellayer, spacialdims []int32, paddingoffset int32, falpha, fbeta float64) (m *DecompressionModule, err error) {
	m = new(DecompressionModule)

	m.module, err = createModule(id, bldr, batch, inputchannel, outputperparallellayer, spacialdims, paddingoffset, falpha, fbeta, true, true)
	return m, err
}

//CompressionModule is a module that concats several layers together when doing the forward and backward passes
type CompressionModule struct {
	*module
}

//CreateCompressionModule will create a simple module with each of the convolution layers being in parallel.
//The parallel convolution will have the same hw, but the channels for each can be changed.
//The number of convolutions depends on the length of the channel array.
//Each convolution will have pad = ((dim-1)*d +1 + offset)/2.  Offset is usually 0, but offset
//can be used to change if the output will be even or odd.  This will make the output for each convolution equal in the spacial dims.
//The strides are stuck at 2.
//
//This considers a stride of 2 spacial dims (hw) need to be odd.
//This will preform a deconvolution with the formula for the output tensor:
//
//N= batch;
//
//C = [neurons[0]+ ... +neurons[i]];
//
//H,W and more (spacial dims) = ((input-1)/2) + 1
//
//the dx tensor is zeroed before back propagation
// If multiple modules share the same input. Each module will need its own tensor as its own dx output. Those tensors need to be summed
//into the output dy tensor of module it got its x input tensor from
func CreateCompressionModule(id int64, bldr *Builder,
	batch, inputchannels int32,
	paralleloutputchans, spacialdims []int32,
	paddingoffset int32,
	falpha, fbeta float64) (m *CompressionModule, err error) {
	m = new(CompressionModule)
	m.module, err = createModule(id, bldr,
		batch, inputchannels, paralleloutputchans,
		spacialdims,
		paddingoffset,
		falpha, fbeta, true, false)
	return m, err
}

//NeutralModule is for nonsliding modules
type NeutralModule struct {
	*module
}

//CreateSingleStridedModule creates a Module with stride set to 1.
//Since there are parallel convolution in this layer the input is shared between all of them.
//the dx tensor is zeroed before back propagation
// If multiple modules share the same input. Each module will need its own tensor as its own dx output. Those tensors need to be summed
//into the output dy tensor of module it got its x input tensor from
func CreateSingleStridedModule(id int64, bldr *Builder,
	batch, inputchannels int32,
	paralleloutputchans, spacialdims []int32,
	paddingoffset int32,
	falpha, fbeta float64, strides, deconv bool) (m *NeutralModule, err error) {
	m = new(NeutralModule)
	m.module, err = createModule(id, bldr,
		batch, inputchannels, paralleloutputchans,
		spacialdims,
		paddingoffset,
		falpha, fbeta, strides, deconv)
	return m, err
}
