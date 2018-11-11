package main

import (
	"fmt"
	"image"
	"math/rand"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/loss"
	"github.com/dereklstinson/GoCuNets/testing/mnist/dfuncs"
	"github.com/dereklstinson/GoCuNets/testing/mnistdcauto/dcnetworks"
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

	handle := cudnn.CreateHandler(devs[0], "/home/derek/go/src/github.com/dereklstinson/GoCudnn/kernels/")
	//stream, err := gocudnn.Cuda{}.CreateBlockingStream()

	//	utils.CheckError(handles.SetStream(stream))
	stream, err := gocudnn.Cuda{}.CreateBlockingStream()
	utils.CheckError(err)
	utils.CheckError(handle.SetStream(stream))
	//Flag managers
	var dataflag cudnn.DataTypeFlag
	var convflag gocudnn.ConvolutionFlags
	var fflag cudnn.TensorFormatFlag

	//Data Locations

	const filedirectory = "../mnist/files/"
	const mnistfilelabel = "train-labels.idx1-ubyte"
	const mnistimage = "train-images.idx3-ubyte"
	const imagesave = "/home/derek/Desktop/DCAutoEncoderGif/"

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
	AutoEncoder := dcnetworks.DcAutoResize(handle, fflag.NCHW(), dataflag.Float(), convflag.Mode.CrossCorrelation(), true, 10)
	//Set the AutoEncoderNetwork hidden layer algo
	utils.CheckError(AutoEncoder.DynamicHidden())

	//Load the batches into gpu mem this is basically the Arabic numbers are place in arabicoutput.T() and arabicnums.DeltaT()
	arabicoutput, arabicnums := putintogpumem(batchesofinputbatches, fflag.NCHW(), dataflag.Float(), []int32{10, 1, 28, 28}, true)

	//Make an imager so we can visually see the progress
	imager, err := imaging.MakeImager(handle)

	utils.CheckError(err)

	//set the number of epocs
	epocs := 100
	snapshotsize := 100
	//Set the Loss Calculator. This is Mean Square Error
	MSE, err := loss.CreateMSECalculatorGPU(handle, true)
	utils.CheckError(err)

	//Need this memory as an inbetween for the Autoencoder and Loss Function so that it can return the errors to the autoencoder
	fconout, err := layers.BuildIO(fflag.NCHW(), dataflag.Float(), []int32{10, 1, 28, 28}, true)
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
			utils.CheckError(AutoEncoder.ForwardProp(handle, nil, arabicnums[j], arabicoutput[j]))
			stream.Sync()
			//Load the outputs from autoencoder into fconout
			fconout.LoadTValues(arabicoutput[j].T().Memer())
			stream.Sync()
			//arabicout contains the the output of the autoencoder in its T() and target values in its DeltaT() fconout will get the errors from the loss function in its DeltaT()
			MSE.ErrorGPU(handle, fconout, arabicoutput[j])
			stream.Sync()
			//MSE.Loss() just returns the loss calculated in MSE.ErrorGPU.  MSE.ErrorGPU doesn't return return the loss it just stores it.
			epocloss += MSE.Loss()
			utils.CheckError(err)
			stream.Sync()
			//BackProp those errors put into fconout back through the auto encoder
			utils.CheckError(AutoEncoder.BackPropFilterData(handle, nil, arabicnums[j], fconout))
			stream.Sync()
			//Update the weights
			utils.CheckError(AutoEncoder.UpdateWeights(handle, 10))
			stream.Sync()

			if j%snapshotsize == 0 {

				utils.CheckError(AutoEncoder.ForwardProp(handle, nil, arabicnums[0], arabicoutput[0]))
				imagerlayer.LoadTValues(arabicoutput[0].T().Memer())
				stream.Sync()
				outputimage, err := imager.TileBatches(handle, imagerlayer, 2, 5)
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

		epocloss /= float32(len(arabicnums))
		stream.Sync()
		fmt.Println("At Epoc: ", i, "Loss is :", epocloss)
		if epocloss <= 11.5 {
			fmt.Println("HIT 11.5 Loss")
			giffer.MakeGrayGif(totalrunimage)
			fmt.Println("Writing GIF")
			utils.CheckError(filing.WritetoHD(imagesave, "AutoDCresize0", giffer))

			//	utils.CheckError(filing.WriteImage(imagesave, "AutoEncoder"+number, outputimage))
			fmt.Println("Done Writing GIF")
			devs[0].Reset()
			return
		}
		//	shuffle(arabicnums, arabicoutput)

	}

	devs[0].Reset()

}
func putintogpumem(arabic [][]float32, frmt cudnn.TensorFormat, dtype cudnn.DataType, dimsarabic []int32, memmanaged bool) (output, runs []*layers.IO) {
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
func shuffle(runs, outputs []*layers.IO) {
	rand.Shuffle(len(runs), func(i, j int) {
		runs[i], runs[j] = runs[j], runs[i]
		outputs[i], outputs[j] = outputs[j], outputs[i]
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
