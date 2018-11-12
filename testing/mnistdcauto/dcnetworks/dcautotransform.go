package dcnetworks

import (
	"fmt"

	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/cnntranspose"
	"github.com/dereklstinson/GoCuNets/trainer"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//DCAutoTransform using the transform method of convtrans
func DCAutoTransform(handle *cudnn.Handler,
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
	CMode gocudnn.ConvolutionMode,
	memmanaged bool,
	batchsize int32) *gocunets.Network {
	in := utils.Dims
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
	const numofneurons = int32(30)
	network.AddLayer(
		cnn.SetupDynamic(handle, frmt, dtype, in(batchsize, 1, 28, 28), filter(numofneurons, 1, 8, 8), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
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
		cnn.SetupDynamic(handle, frmt, dtype, in(batchsize, numofneurons, 21, 21), filter(numofneurons, numofneurons, 8, 8), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)
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
		cnn.SetupDynamic(handle, frmt, dtype, in(batchsize, numofneurons, 14, 14), filter(numofneurons, numofneurons, 8, 8), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	) // 14-8+1 =7
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
		cnn.SetupDynamic(handle, frmt, dtype, in(batchsize, numofneurons, 7, 7), filter(4, numofneurons, 7, 7), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	) //7-7+1
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
		cnn.SetupDynamic(handle, frmt, dtype, in(batchsize, 4, 1, 1), filter(numofneurons, 4, 7, 7), CMode, padding(6, 6), stride(1, 1), dilation(1, 1), memmanaged),
	) //1-5+(2*4) +1 = 5
	/*
		Activation Layer D2      10
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)
	/*
		Convoultion Layer D3       11
	*/
	network.AddLayer(
		cnntranspose.Transform(handle, frmt, dtype, in(batchsize, numofneurons, 14, 14), filter(numofneurons, numofneurons, 7, 7), CMode, padding(3, 3), stride(1, 1), dilation(1, 1), memmanaged),
	)
	/*
		Activation Layer D4        13
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)

	/*
		Convoultion Layer D5       14
	*/
	network.AddLayer(
		cnntranspose.Transform(handle, frmt, dtype, in(batchsize, numofneurons, 28, 28), filter(numofneurons, numofneurons, 7, 7), CMode, padding(3, 3), stride(1, 1), dilation(1, 1), memmanaged),
	) //28 -7 + 1 +3*2 = 28
	/*
		Activation Layer D6        15
	*/
	network.AddLayer(
		activation.Leaky(handle),
	)

	/*
		Convoultion Layer D7         16
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle, frmt, dtype, in(batchsize, numofneurons, 28, 28), filter(1, numofneurons, 7, 7), CMode, padding(3, 3), stride(1, 1), dilation(1, 1), memmanaged),
	) //28 -7 + 1 +3*2 = 28

	//var err error
	numoftrainers := network.TrainersNeeded()
	fmt.Println("Trainers Needed", numoftrainers)
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
