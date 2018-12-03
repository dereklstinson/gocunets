package dcnetworks

import (
	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/trainer"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//DcAutoNoConvTrans using regular method of increasing size of convolution...by just increasing the outer padding
func DcAutoNoConvTrans(handle *cudnn.Handler,
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
	CMode gocudnn.ConvolutionMode,
	memmanaged bool,
	batchsize int32) *gocunets.Network {

	filter := utils.Dims
	padding := utils.Dims
	stride := utils.Dims
	dilation := utils.Dims
	//var tmdf gocudnn.TrainingModeFlag
	//tmode := tmdf.Adam()
	//var aflg gocudnn.ActivationModeFlag

	network := gocunets.CreateNetwork()
	//Setting Up Network

	/*
		Convoultion Layer E1  1
	*/
	const numofneurons = int32(50)
	network.AddLayer(
		cnn.SetupDynamic(handle, frmt, dtype, filter(numofneurons, 1, 8, 8), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	) //28-8+1 = 21
	/*
		Activation Layer E2    2
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)
	/*
		Convoultion Layer E3    3
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle, frmt, dtype, filter(numofneurons, numofneurons, 8, 8), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	) //21-8+1 =14
	/*
		Activation Layer E4    4
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)

	/*
		Convoultion Layer E5    5
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle, frmt, dtype, filter(numofneurons, numofneurons, 8, 8), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	) // 14-8+1=7
	/*
		Activation Layer E6    6
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)
	/*
		Convoultion Layer E7    7
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle, frmt, dtype, filter(4, numofneurons, 7, 7), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	) // 1

	/*
		Activation Layer MIDDLE    8
	*/
	network.AddLayer(

		activation.Leaky(handle),
	)

	/*
		Convoultion Layer D1       9
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle, frmt, dtype, filter(numofneurons, 4, 7, 7), CMode, padding(6, 6), stride(1, 1), dilation(1, 1), memmanaged),
	) //7
	/*
		Activation Layer D2       10
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)
	/*
		Convoultion Layer D3      11
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle, frmt, dtype, filter(numofneurons, numofneurons, 8, 8), CMode, padding(7, 7), stride(1, 1), dilation(1, 1), memmanaged),
	) //7-8+(14)+1 =14
	/*
		Activation Layer D4        12
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)

	/*
		Convoultion Layer D5       13
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle, frmt, dtype, filter(numofneurons, numofneurons, 8, 8), CMode, padding(7, 7), stride(1, 1), dilation(1, 1), memmanaged),
	) //14-8 +14 +1 =21
	/*
		Activation Layer D6       14
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)

	/*
		Convoultion Layer D7         15
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle, frmt, dtype, filter(1, numofneurons, 8, 8), CMode, padding(7, 7), stride(1, 1), dilation(1, 1), memmanaged),
	) //28

	//var err error
	numoftrainers := network.TrainersNeeded()

	trainersbatch := make([]trainer.Trainer, numoftrainers) //If these were returned then you can do some training parameter adjustements on the fly
	trainerbias := make([]trainer.Trainer, numoftrainers)   //If these were returned then you can do some training parameter adjustements on the fly
	for i := 0; i < numoftrainers; i++ {
		a, b, err := trainer.SetupAdamWandB(handle.XHandle(), .00001, .00001, batchsize)
		a.SetRate(.001) //This is here to change the rate if you so want to
		b.SetRate(.001)

		trainersbatch[i], trainerbias[i] = a, b

		if err != nil {
			panic(err)
		}

	}
	network.LoadTrainers(handle, trainersbatch, trainerbias) //Load the trainers in the order they are needed
	return network
}
