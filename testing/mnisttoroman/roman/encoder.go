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
	var err error
	numoftrainers := network.TrainersNeeded()
	trainersbatch := make([]trainer.Trainer, numoftrainers)
	trainerbias := make([]trainer.Trainer, numoftrainers)
	for i := 0; i < numoftrainers; i++ {
		//	a, b, err1 :=
		//	a.SetRate(.001)
		//	b.SetRate(.001)
		trainersbatch[i], trainerbias[i], err = trainer.SetupAdamWandB(handle.XHandle(), .000001, .00001, batchsize) //a, b, err1

		if err != nil {
			panic(err)
		}

	}
	network.LoadTrainers(handle, trainersbatch, trainerbias)
	return network
}
