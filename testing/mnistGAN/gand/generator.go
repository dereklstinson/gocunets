package gand

import (
	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/xactivation"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Generator returns a generator network
func Generator(handle *gocunets.Handles, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, CMode gocudnn.ConvolutionMode, memmanaged bool, batchsize int) *gocunets.Network {
	in := dims
	filter := dims
	padding := dims
	stride := dims
	dilation := dims
	var aflg gocudnn.ActivationModeFlag

	network := gocunets.CreateNetwork()
	//Setting Up Network
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 1, 28, 28), filter(20, 1, 5, 5), CMode, padding(2, 2), stride(1, 1), dilation(1, 1), memmanaged),
	)
	network.AddLayer( //activation
		xactivation.SetupLeaky(handle.XHandle(), dtype),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 20, 28, 28), filter(20, 20, 5, 5), CMode, padding(2, 2), stride(1, 1), dilation(1, 1), memmanaged),
	)
	network.AddLayer( //activation
		xactivation.SetupLeaky(handle.XHandle(), dtype),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 20, 28, 28), filter(20, 20, 5, 5), CMode, padding(2, 2), stride(1, 1), dilation(1, 1), memmanaged),
	)
	network.AddLayer( //activation
		xactivation.SetupLeaky(handle.XHandle(), dtype),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 20, 28, 28), filter(1, 20, 5, 5), CMode, padding(2, 2), stride(1, 1), dilation(1, 1), memmanaged),
	)
	network.AddLayer( //activation
		activation.Setup(aflg.Tanh()),
	)
	return network
}
