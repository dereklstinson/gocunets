package main

import (
	"fmt"
	"image"
	"math/rand"

	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/activation"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/xactivation"
	"github.com/dereklstinson/GoCuNets/loss"
	"github.com/dereklstinson/GoCuNets/testing/mnist/dfuncs"
	"github.com/dereklstinson/GoCuNets/trainer"
	"github.com/dereklstinson/GoCuNets/utils"
	"github.com/dereklstinson/GoCuNets/utils/filing"
	"github.com/dereklstinson/GoCuNets/utils/imaging"
	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/nfnt/resize"
)

func main() {

	network()
}
func network() {
	//Setting up GPU and Handles and Steams
	gocudnn.Cuda{}.LockHostThread()
	devs, err := gocudnn.Cuda{}.GetDeviceList()
	utils.CheckError(err)
	utils.CheckError(devs[0].Set())
	handles := gocunets.CreateHandle(devs[0], "/home/derek/go/src/github.com/dereklstinson/GoCudnn/kernels/")
	stream, err := gocudnn.Cuda{}.CreateBlockingStream()
	utils.CheckError(err)
	utils.CheckError(handles.SetStream(stream))

	//Flag managers
	var dataflag gocudnn.DataTypeFlag
	var convflag gocudnn.ConvolutionFlags
	var fflag gocudnn.TensorFormatFlag

	//Data Locations

	const filedirectory = "../mnist/files/"
	const mnistfilelabel = "train-labels.idx1-ubyte"
	const mnistimage = "train-images.idx3-ubyte"
	const imagesave = "/home/derek/Desktop/AutoEncoderGif/"

	//Load the mnist data
	mnistdata, err := dfuncs.LoadMNIST(filedirectory, mnistfilelabel, mnistimage)
	utils.CheckError(err)

	//Normalize The Data
	avg := dfuncs.FindAverage(mnistdata)
	mnistdata = dfuncs.NormalizeData(mnistdata, avg)

	//Organize the batches into batches of 0 to 9 so that batchsize will be 10
	sectioned := makenumbers(mnistdata)

	//Make the batch up the batches.  this would be number of runs for an epoc
	batchesofinputbatches := makeinputbatches(sectioned)
	fmt.Println("Number of Runs: ", len(batchesofinputbatches))

	//Make Autoencoder network
	AutoEncoder := encoder(handles, fflag.NCHW(), dataflag.Float(), convflag.Mode.CrossCorrelation(), true, 10)
	//Set the AutoEncoderNetwork hidden layer algo
	utils.CheckError(AutoEncoder.DynamicHidden())

	//Load the batches into gpu mem this is basically the Arabic numbers are place in arabicoutput.T() and arabicnums.DeltaT()
	arabicoutput, arabicnums := putintogpumem(batchesofinputbatches, fflag.NCHW(), dataflag.Float(), []int32{10, 28 * 28, 1, 1}, true)

	//Make an imager so we can visually see the progress
	imager, err := imaging.MakeImager(handles.XHandle())

	utils.CheckError(err)

	//set the number of epocs
	epocs := 100
	snapshotsize := 400
	//Set the Loss Calculator. This is Mean Square Error
	MSE, err := loss.CreateMSECalculatorGPU(handles.XHandle(), true)
	utils.CheckError(err)

	//Need this memory as an inbetween for the Autoencoder and Loss Function so that it can return the errors to the autoencoder
	fconout, err := layers.BuildIO(fflag.NCHW(), dataflag.Float(), []int32{10, 784, 1, 1}, true)
	utils.CheckError(err)
	//Need this to reshape the output of the autoencoder into something the imager can use to make an image.Image
	imagerlayer, err := layers.BuildIO(fflag.NCHW(), dataflag.Float(), []int32{10, 1, 28, 28}, true)
	utils.CheckError(err)
	totalrunimage := make([]image.Image, 0)
	for i := 0; i < epocs; i++ {
		giffer := imaging.NewGiffer(0, 1) //giffer stacks a bunch of images and puts them into a gif
		images := make([]image.Image, 0)
		//Making a lossaray to calculate the loss per batch
		epocloss := float32(0)
		for j := range arabicnums {
			stream.Sync()
			utils.CheckError(AutoEncoder.ForwardProp(handles, nil, arabicnums[j], arabicoutput[j]))
			stream.Sync()
			//Load the outputs from autoencoder into fconout
			fconout.LoadTValues(arabicoutput[j].T().Memer())
			stream.Sync()
			//arabicout contains the the output of the autoencoder in its T() and target values in its DeltaT() fconout will get the errors from the loss function in its DeltaT()
			MSE.ErrorGPU(handles.XHandle(), fconout, arabicoutput[j])
			stream.Sync()
			//MSE.Loss() just returns the loss calculated in MSE.ErrorGPU.  MSE.ErrorGPU doesn't return return the loss it just stores it.
			epocloss += MSE.Loss()
			utils.CheckError(err)
			stream.Sync()
			//BackProp those errors put into fconout back through the auto encoder
			utils.CheckError(AutoEncoder.BackPropFilterData(handles, nil, arabicnums[j], fconout))
			stream.Sync()
			//Update the weights
			utils.CheckError(AutoEncoder.UpdateWeights(handles, 10))
			stream.Sync()
			imagerlayer.LoadTValues(fconout.T().Memer())
			stream.Sync()
			if j%snapshotsize == 0 {
				outputimage, err := imager.TileBatches(handles.XHandle(), imagerlayer, 2, 5)
				utils.CheckError(err)
				images = append(images, outputimage)
				//	fmt.Println("Grabbing Image:", j)
				stream.Sync()
			}

		}
		somenewimages := make([]image.Image, len(images))
		for j := range images {
			somenewimages[j] = resize.Resize(0, 280, images[j], resize.NearestNeighbor)
		}
		totalrunimage = append(totalrunimage, somenewimages...)
		//	fmt.Println("MakingGif: Start")

		//	fmt.Println("MakingGif: Done")
		//	outputimage = resize.Resize(0, 280, outputimage, resize.NearestNeighbor)
		//giffer.Append(outputimage)

		//Load the values from the autoencoder into the imagerlayer so we can print those dang numbers

		//Tile those numbers into a 2 by 5 output

		//This makes the number into a file appropriate numbers to keep the order like 0001 and 0002
		//number := utils.NumbertoString(i, epocs)
		epocloss /= float32(len(arabicnums))
		stream.Sync()
		fmt.Println("At Epoc: ", i, "Loss is :", epocloss)
		if epocloss <= 13 {
			fmt.Println("HIT 13 Loss")
			giffer.MakeGrayGif(totalrunimage)
			fmt.Println("Writing GIF")
			utils.CheckError(filing.WritetoHD(imagesave, "AutoGifsToLoss13", giffer))

			//	utils.CheckError(filing.WriteImage(imagesave, "AutoEncoder"+number, outputimage))
			fmt.Println("Done Writing GIF")
			fmt.Println("Should Break Out of Loop")
			break
		}

	}
	fmt.Println("BrokeOut --- BYE")
	devs[0].Reset()

}
func putintogpumem(arabic [][]float32, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dimsarabic []int32, memmanaged bool) (output, runs []*layers.IO) {
	var err error
	runs = make([]*layers.IO, len(arabic))
	output = make([]*layers.IO, len(arabic))
	for i := range arabic {
		runs[i], err = layers.BuildNetworkInputIO(frmt, dtype, dimsarabic, memmanaged)
		utils.CheckError(err)
		ptr, err := gocudnn.MakeGoPointer(arabic[i])
		utils.CheckError(err)
		utils.CheckError(runs[i].LoadTValues(ptr))
		output[i], err = layers.BuildIO(frmt, dtype, dimsarabic, memmanaged)
		utils.CheckError(err)
		utils.CheckError(output[i].LoadDeltaTValues(ptr))
	}
	return output, runs
}
func shuffle(runs []*layers.IO) {
	rand.Shuffle(len(runs), func(i, j int) {
		runs[i], runs[j] = runs[j], runs[i]
	})
}

func makeinputbatches(sections []number) [][]float32 {
	min := int(9999999)
	for i := range sections {
		if min > len(sections[i].mnist) {
			min = len(sections[i].mnist)
		}
	}
	numofbatches := min
	fmt.Println(numofbatches)
	numinbatches := len(sections)
	fmt.Println(numinbatches)
	batches := make([][]float32, numofbatches)
	imgsize := 28 * 28
	for i := range batches {
		batches[i] = make([]float32, numinbatches*imgsize)
	}
	for i := range sections {
		for j := 0; j < numofbatches; j++ {
			for k := range sections[i].mnist[j].Data {

				batches[j][i*imgsize+k] = sections[i].mnist[j].Data[k]
			}

		}

	}
	return batches
}

func makenumbers(mnist []dfuncs.LabeledData) []number {
	sections := make([]number, 10)
	for i := range mnist {
		nmbr := mnist[i].Number
		sections[nmbr].mnist = append(sections[nmbr].mnist, mnist[i])
	}
	return sections
}

type number struct {
	mnist []dfuncs.LabeledData
}

//Encoder is sort of like an auto encoder
func encoder(handle *gocunets.Handles, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, CMode gocudnn.ConvolutionMode, memmanaged bool, batchsize int32) *gocunets.Network {
	in := utils.Dims
	filter := utils.Dims
	padding := utils.Dims
	stride := utils.Dims
	dilation := utils.Dims

	var aflg gocudnn.ActivationModeFlag

	network := gocunets.CreateNetwork()
	//Setting Up Network
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 784, 1, 1), filter(50, 784, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	) //28-15+14
	network.AddLayer( //activation
		xactivation.SetupLeaky(handle.XHandle(), dtype),
	//	activation.Setup(aflg.Tanh()),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 50, 1, 1), filter(50, 50, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	) //28-15+14
	network.AddLayer( //activation
		xactivation.SetupLeaky(handle.XHandle(), dtype),
	//	activation.Setup(aflg.Tanh()),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 50, 1, 1), filter(4, 50, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)
	network.AddLayer( //activation
		//xactivation.SetupLeaky(handle.XHandle(), dtype),
		activation.Setup(aflg.Sigmoid()),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 4, 1, 1), filter(50, 4, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	)
	network.AddLayer( //activation
		xactivation.SetupLeaky(handle.XHandle(), dtype),
	//	activation.Setup(aflg.Tanh()),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 50, 1, 1), filter(50, 50, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
	) //28-15+14
	network.AddLayer( //activation
		xactivation.SetupLeaky(handle.XHandle(), dtype),
		//activation.Setup(aflg.Tanh()),
	)
	network.AddLayer( //convolution
		cnn.SetupDynamic(handle.Cudnn(), frmt, dtype, in(batchsize, 50, 1, 1), filter(784, 50, 1, 1), CMode, padding(0, 0), stride(1, 1), dilation(1, 1), memmanaged),
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
		a, b, err := trainer.SetupAdamWandB(handle.XHandle(), .000001, .000001, batchsize)
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
