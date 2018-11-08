package networks

import (
	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/xactivation"
	"github.com/dereklstinson/GoCuNets/trainer"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//ParaChanAuto is sort of like an auto encoder
func ParaChanAuto(handle *gocunets.Handles,
	frmt gocudnn.TensorFormat,
	dtype gocudnn.DataType,
	CMode gocudnn.ConvolutionMode,
	memmanaged bool,
	batchsize int32) *gocunets.Network {
	in := utils.Dims
	filter := utils.Dims
	padding := utils.Dims
	stride := utils.Dims
	dilation := utils.Dims
	var tmdf gocudnn.TrainingModeFlag
	tmode := tmdf.Adam()
	var aflg gocudnn.ActivationModeFlag

	network := gocunets.CreateNetwork()
	//Setting Up Network

	/*
		Convoultion Layer E1
	*/
	const numofneurons = int32(120)
	network.AddLayer(
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 784, 1, 1), filter(50, 784, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)
	/*
		Activation Layer E2
	*/
	network.AddLayer(
		xactivation.SetupAdvanceThreshRelu(handle.XHandle(), in(batchsize, 50, 1, 1), frmt, dtype, tmode, memmanaged),
	)
	/*
		Convoultion Layer E3
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 50, 1, 1), filter(numofneurons, 50, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)
	/*
		Activation Layer E4
	*/
	network.AddLayer(
		xactivation.SetupAdvanceThreshRelu(handle.XHandle(), in(batchsize, numofneurons, 1, 1), frmt, dtype, tmode, memmanaged),
	)

	/*
		Convoultion Layer E5
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, numofneurons, 1, 1), filter(numofneurons, numofneurons, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)
	/*
		Activation Layer E6
	*/
	network.AddLayer(
		xactivation.SetupAdvanceThreshRelu(handle.XHandle(), in(batchsize, numofneurons, 1, 1), frmt, dtype, tmode, memmanaged),
	)
	/*
		Convoultion Layer E7
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, numofneurons, 1, 1), filter(numofneurons, numofneurons, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)
	/*
		Activation Layer E8
	*/
	network.AddLayer(
		xactivation.SetupAdvanceThreshRelu(handle.XHandle(), in(batchsize, numofneurons, 1, 1), frmt, dtype, tmode, memmanaged),
	)
	/*
		Convoultion Layer E9
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, numofneurons, 1, 1), filter(numofneurons, numofneurons, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)

	/*
		Activation Layer E10
	*/
	network.AddLayer(
		xactivation.SetupAdvanceThreshRelu(handle.XHandle(), in(batchsize, numofneurons, 1, 1), frmt, dtype, tmode, memmanaged),
	)
	/*
		Convoultion Layer E11
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, numofneurons, 1, 1), filter(4, numofneurons, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)
	/*
		Activation Layer MIDDLE
	*/
	network.AddLayer(

		activation.Setup(aflg.Sigmoid()),
	)

	/*
		Convoultion Layer D1
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 4, 1, 1), filter(numofneurons, 4, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)
	/*
		Activation Layer D2
	*/
	network.AddLayer(
		xactivation.SetupAdvanceThreshRelu(handle.XHandle(), in(batchsize, numofneurons, 1, 1), frmt, dtype, tmode, memmanaged),
	)
	/*
		Convoultion Layer D3
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, numofneurons, 1, 1), filter(numofneurons, numofneurons, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)
	/*
		Activation Layer D4
	*/
	network.AddLayer(
		xactivation.SetupAdvanceThreshRelu(handle.XHandle(), in(batchsize, numofneurons, 1, 1), frmt, dtype, tmode, memmanaged),
	)

	/*
		Convoultion Layer D5
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, numofneurons, 1, 1), filter(numofneurons, numofneurons, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)
	/*
		Activation Layer D6
	*/
	network.AddLayer(
		xactivation.SetupAdvanceThreshRelu(handle.XHandle(), in(batchsize, numofneurons, 1, 1), frmt, dtype, tmode, memmanaged),
	)

	/*
		Convoultion Layer D7
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, numofneurons, 1, 1), filter(numofneurons, numofneurons, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)
	/*
		Activation Layer D8
	*/
	network.AddLayer(
		xactivation.SetupAdvanceThreshRelu(handle.XHandle(), in(batchsize, numofneurons, 1, 1), frmt, dtype, tmode, memmanaged),
	)
	/*
		Convoultion Layer D9
	*/
	network.AddLayer(
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, numofneurons, 1, 1), filter(numofneurons, numofneurons, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)
	/*
		Activation Layer D10
	*/
	network.AddLayer(
		xactivation.SetupAdvanceThreshRelu(handle.XHandle(), in(batchsize, numofneurons, 1, 1), frmt, dtype, tmode, memmanaged),
	)
	/*
		Convoultion Layer D11
	*/
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, numofneurons, 1, 1), filter(784, numofneurons, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)
	/*
		network.AddLayer( //activation
			activation.Setup(aflg.Tanh()),
		)
	*/
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
