package roman

import (
	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/xactivation"
	"github.com/dereklstinson/GoCuNets/trainer"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Encoder is sort of like an auto encoder
func Encoder(handle *gocunets.Handles, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, CMode gocudnn.ConvolutionMode, memmanaged bool, batchsize int32) *gocunets.Network {
	in := utils.Dims
	filter := utils.Dims
	padding := utils.Dims
	stride := utils.Dims
	dilation := utils.Dims
	var aflg gocudnn.ActivationModeFlag

	network := gocunets.CreateNetwork()
	//Setting Up Network
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 1, 28, 28), filter(12, 1, 13, 13), CMode, padding(6, 6), stride(1, 1), dilation(1, 1), memmanaged),
	) //28-15+14
	network.AddLayer( //activation
		xactivation.SetupLeaky(handle.XHandle(), dtype),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 12, 28, 28), filter(12, 12, 5, 5), CMode, padding(2, 2), stride(1, 1), dilation(1, 1), memmanaged),
	)
	network.AddLayer( //activation
		xactivation.SetupLeaky(handle.XHandle(), dtype),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 12, 28, 28), filter(12, 12, 5, 5), CMode, padding(2, 2), stride(1, 1), dilation(1, 1), memmanaged),
	)
	network.AddLayer( //activation
		xactivation.SetupLeaky(handle.XHandle(), dtype),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 12, 28, 28), filter(1, 12, 5, 5), CMode, padding(2, 2), stride(1, 1), dilation(1, 1), memmanaged),
	)
	network.AddLayer( //activation
		activation.Setup(aflg.Tanh()),
	)
	//var err error
	numoftrainers := network.TrainersNeeded()
	trainersbatch := make([]trainer.Trainer, numoftrainers)
	trainerbias := make([]trainer.Trainer, numoftrainers)
	for i := 0; i < numoftrainers; i++ {
		a, b, err := trainer.SetupAdamWandB(handle.XHandle(), .000001, .00001, batchsize)
		a.SetRate(1)
		b.SetRate(1)

		trainersbatch[i], trainerbias[i] = a, b //a, b, err1

		//trainersbias[i].SetRate(1.0)
		if err != nil {
			panic(err)
		}

	}
	network.LoadTrainers(handle, trainersbatch, trainerbias)
	return network
}
