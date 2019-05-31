package dcnetworks

import (
	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/cnntranspose"
	"github.com/dereklstinson/GoCuNets/trainer"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//DCAutoReverse using regular method of increasing size of convolution...by just increasing the outer padding
func DCAutoReverse(handle *cudnn.Handler,
	frmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
	CMode gocudnn.ConvolutionMode,
	memmanaged bool,
	batchsize int32) *gocunets.Network {

	filter := utils.Dims
	padding := utils.Dims
	stride := utils.Dims
	dilation := utils.Dims
	//var tmdf gocudnn.TrainingModeFlag
	//tmode := tmdf.Adam()
	//var aflg gocudnn.ActivationModeFlag //
	var cflg gocudnn.ConvolutionMode
	reversecmode := cflg.Convolution()
	network := gocunets.CreateNetwork()
	//Setting Up Network

	/*
		Convoultion Layer E1  0
	*/
	const numofneurons = int32(30)
	network.AddLayer(
		cnn.Setup(handle, frmt, dtype, filter(numofneurons, 1, 8, 8), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), 1),
	) //28-8+1 = 21
	/*
		Activation Layer E2    1
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)
	/*
		Convoultion Layer E3    2
	*/
	network.AddLayer(
		cnn.Setup(handle, frmt, dtype, filter(numofneurons, numofneurons, 8, 8), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), 2),
	) //21-8+1 =14
	/*
		Activation Layer E4    3
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)

	/*
		Convoultion Layer E5    4
	*/
	network.AddLayer(
		cnn.Setup(handle, frmt, dtype, filter(numofneurons, numofneurons, 8, 8), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), 3),
	) // 14-8+1=7
	/*
		Activation Layer E6    5
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)
	/*
		Convoultion Layer E7    6
	*/
	network.AddLayer(
		cnn.Setup(handle, frmt, dtype, filter(4, numofneurons, 7, 7), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), 4),
	) // 1

	/*
		Activation Layer MIDDLE    7
	*/
	network.AddLayer(

		activation.Leaky(handle),
	)

	/*
		Convoultion Layer D1       8
	*/
	network.AddLayer(
		cnntranspose.ReverseBuild(handle, frmt, dtype, filter(4, numofneurons, 7, 7), reversecmode, padding(0, 0), stride(1, 1), dilation(1, 1), false, 5),
	) //7
	/*
		Activation Layer D2       9
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)
	/*
		Convoultion Layer D3      10
	*/
	network.AddLayer(
		cnntranspose.ReverseBuild(handle, frmt, dtype, filter(numofneurons, numofneurons, 8, 8), reversecmode, padding(0, 0), stride(1, 1), dilation(1, 1), false, 6),
	) //7-8+(14)+1 =14
	/*
		Activation Layer D4        11
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)

	/*
		Convoultion Layer D5       12
	*/
	network.AddLayer(
		cnntranspose.ReverseBuild(handle, frmt, dtype, filter(numofneurons, numofneurons, 8, 8), reversecmode, padding(0, 0), stride(1, 1), dilation(1, 1), false, 7),
	) //14-8 +14 +1 =21
	/*
		Activation Layer D6       13
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)

	/*
		Convoultion Layer D7         14
	*/
	network.AddLayer(
		cnntranspose.ReverseBuild(handle, frmt, dtype, filter(numofneurons, 1, 8, 8), reversecmode, padding(0, 0), stride(1, 1), dilation(1, 1), false, 8),
	) //28

	//var err error
	numoftrainers := network.TrainersNeeded()

	trainersbatch := make([]trainer.Trainer, numoftrainers) //If these were returned then you can do some training parameter adjustements on the fly
	trainerbias := make([]trainer.Trainer, numoftrainers)   //If these were returned then you can do some training parameter adjustements on the fly
	for i := 0; i < numoftrainers; i++ {
		a, b, err := trainer.SetupAdamWandB(handle.XHandle(), .00001, .00001, batchsize)
		a.SetRates(.001, .001) //This is here to change the rate if you so want to
		b.SetRates(.001, .001)

		trainersbatch[i], trainerbias[i] = a, b

		if err != nil {
			panic(err)
		}

	}
	network.LoadTrainers(handle, trainersbatch, trainerbias) //Load the trainers in the order they are needed
	return network
}
