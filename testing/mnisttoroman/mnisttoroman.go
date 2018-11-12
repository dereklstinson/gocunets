package main

import (
	"fmt"
	"image"
	"math/rand"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/loss"
	"github.com/dereklstinson/GoCuNets/testing/mnist/dfuncs"
	"github.com/dereklstinson/GoCuNets/testing/mnisttoroman/roman"
	"github.com/dereklstinson/GoCuNets/utils"
	"github.com/dereklstinson/GoCuNets/utils/filing"
	"github.com/dereklstinson/GoCuNets/utils/imaging"
	"github.com/dereklstinson/GoCudnn"
	"github.com/nfnt/resize"
)

func main() {
	gocudnn.Cuda{}.LockHostThread()
	const romanimagelocation = "../mnist/roman/"
	const filedirectory = "../mnist/files/"
	const mnistfilelabel = "train-labels.idx1-ubyte"
	const mnistimage = "train-images.idx3-ubyte"
	const imagesave = "/home/derek/Desktop/RomanOutput/"
	romannums := roman.GetRoman(romanimagelocation)
	mnistdata, err := dfuncs.LoadMNIST(filedirectory, mnistfilelabel, mnistimage)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(romannums))
	fmt.Println(len(mnistdata))
	avg := dfuncs.FindAverage(mnistdata)
	mnistdata = dfuncs.NormalizeData(mnistdata, avg)
	romannums = roman.Normalize(romannums, avg) //I am not doing a total average just a psudo for this.
	//  There is a 6000/1 mnistdata to roman on this.  It won't make that much of a difference
	sectioned := makenumbers(romannums, mnistdata)
	batchedoutput := makeoutputbatch(sectioned)
	batchesofinputbatches := makeinputbatches(sectioned)
	fmt.Println(len(batchedoutput)) //batch is 10 chanel is 1 and dims are 28 by 28
	fmt.Println(len(batchesofinputbatches))
	//runsize := len(batchesofinputbatches)

	devs, err := gocudnn.Cuda{}.GetDeviceList()
	utils.CheckError(err)
	handles := cudnn.CreateHandler(devs[0], "/home/derek/go/src/github.com/dereklstinson/GoCudnn/kernels/")
	stream, err := gocudnn.Cuda{}.CreateBlockingStream()
	utils.CheckError(err)
	handles.SetStream(stream)
	var dataflag cudnn.DataTypeFlag
	var convflag gocudnn.ConvolutionFlags
	var fflag cudnn.TensorFormatFlag
	toroman := roman.Encoder(handles, fflag.NCHW(), dataflag.Float(), convflag.Mode.CrossCorrelation(), true, 10)
	utils.CheckError(toroman.DynamicHidden())
	romanoutput, arabicnums := putintogpumem(batchedoutput, []int32{10, 1, 28, 28}, batchesofinputbatches, fflag.NCHW(), dataflag.Float(), []int32{10, 1, 28, 28}, true)
	imager, err := imaging.MakeImager(handles)
	utils.CheckError(err)
	epocs := 100
	snapshotsize := 100
	MSE, err := loss.CreateMSECalculatorGPU(handles, true)
	utils.CheckError(err)

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
			utils.CheckError(toroman.ForwardProp(handles, nil, arabicnums[j], romanoutput))
			stream.Sync()
			fconout.LoadTValues(romanoutput.T().Memer())
			MSE.ErrorGPU(handles, fconout, romanoutput)
			utils.CheckError(err)
			stream.Sync()
			epocloss += MSE.Loss()
			stream.Sync()
			utils.CheckError(toroman.BackPropFilterData(handles, nil, arabicnums[j], fconout))
			stream.Sync()
			utils.CheckError(toroman.UpdateWeights(handles, 10))
			stream.Sync()
			if j%snapshotsize == 0 {

				utils.CheckError(toroman.ForwardProp(handles, nil, arabicnums[0], romanoutput))
				imagerlayer.LoadTValues(fconout.T().Memer())
				stream.Sync()
				outputimage, err := imager.TileBatches(handles, imagerlayer, 2, 5)
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
		if epocloss <= 10.5 || i >= 20 {
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

func putintogpumem(romans []float32, dimsroman []int32, arabic [][]float32, frmt cudnn.TensorFormat, dtype cudnn.DataType, dimsarabic []int32, memmanaged bool) (output *layers.IO, runs []*layers.IO) {
	var err error
	runs = make([]*layers.IO, len(arabic))
	for i := range arabic {
		runs[i], err = layers.BuildNetworkInputIO(frmt, dtype, dimsarabic, memmanaged)
		utils.CheckError(err)
		ptr, err := gocudnn.MakeGoPointer(arabic[i])
		utils.CheckError(err)
		utils.CheckError(runs[i].LoadTValues(ptr))

	}
	output, err = layers.BuildNetworkOutputIOFromSlice(frmt, dtype, dimsroman, memmanaged, romans)
	utils.CheckError(err)
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
func makeoutputbatch(sections []number) []float32 {
	batches := make([]float32, len(sections)*28*28)
	for i := range sections {
		for j := range sections[i].roman.Data {
			batches[i*28*28+j] = sections[i].roman.Data[j]
		}
	}
	return batches
}

func makenumbers(rmn []roman.Roman, mnist []dfuncs.LabeledData) []number {
	sections := make([]number, len(rmn))
	for i := range mnist {
		nmbr := mnist[i].Number
		sections[nmbr].mnist = append(sections[nmbr].mnist, mnist[i])
	}
	for i := range rmn {
		nmbr := rmn[i].Number
		sections[nmbr].roman = rmn[i]
	}
	return sections
}

type number struct {
	mnist []dfuncs.LabeledData
	roman roman.Roman
}
