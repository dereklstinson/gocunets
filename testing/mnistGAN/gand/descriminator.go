package gand

import (
	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/fcnn"
	"github.com/dereklstinson/GoCuNets/layers/softmax"
	"github.com/dereklstinson/GoCuNets/layers/xactivation"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

func Descriminator(handle *gocunets.Handles, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, CMode gocudnn.ConvolutionMode, memmanaged bool, batchsize int) *gocunets.Network {
	in := dims
	filter := dims
	padding := dims
	stride := dims
	dilation := dims
	//	var aflg gocudnn.ActivationModeFlag

	network := gocunets.CreateNetwork()
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 1, 28, 28), filter(20, 1, 6, 6), CMode, padding(2, 2), stride(2, 2), dilation(1, 1), memmanaged),
	) // some math    (28-6+4)/2 = 13 , 13 +1 =14,
	network.AddLayer( //activation
		xactivation.SetupLeaky(handle.XHandle(), dtype),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 20, 14, 14), filter(20, 20, 6, 6), CMode, padding(2, 2), stride(2, 2), dilation(1, 1), memmanaged),
	) // some math (14-6+4)/2 = 6, 6+1=7
	network.AddLayer( //activation
		xactivation.SetupLeaky(handle.XHandle(), dtype),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 20, 7, 7), filter(20, 20, 5, 5), CMode, padding(1, 1), stride(1, 1), dilation(1, 1), memmanaged),
	) //some match (7-5+2)/1=4, 4+1=5
	network.AddLayer( //activation
		xactivation.SetupLeaky(handle.XHandle(), dtype),
	)
	network.AddLayer(
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 20, 7, 7), filter(20, 20, 5, 5), CMode, padding(1, 1), stride(1, 1), dilation(1, 1), memmanaged),
	) //some match (5-5+2)/1=2, 2+1=3
	network.AddLayer( //activation
		xactivation.SetupLeaky(handle.XHandle(), dtype),
	)
	network.AddLayer( //convolution
		fcnn.CreateFromshapeNoOut(handle.Cudnn(), 2, in(batchsize, 20, 3, 3), memmanaged, dtype, frmt),
	)
	network.AddLayer( //softmaxoutput
		softmax.BuildNoErrorChecking(), nil,
	)
	return network
}
