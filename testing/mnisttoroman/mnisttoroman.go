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
	"github.com/dereklstinson/GoCuNets/ui"
	"github.com/dereklstinson/GoCuNets/utils"
	"github.com/dereklstinson/GoCuNets/utils/filing"
	"github.com/dereklstinson/GoCuNets/utils/imaging"
	"github.com/dereklstinson/GoCudnn"
	"github.com/nfnt/resize"
)

const learningrates = .001
const codingvector = int32(10)
const numofneurons = int32(50)
const l1regularization = float32(.00001)
const l2regularization = float32(.00001)
const metabatchsize = 2
const buffersize = 5000

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
	Encoder := roman.ArabicEncoder(handles, fflag.NCHW(), dataflag.Float(), convflag.Mode.CrossCorrelation(), true, 10, int32(metabatchsize*10), learningrates, codingvector, numofneurons, l1regularization, l2regularization)
	utils.CheckError(Encoder.DynamicHidden())
	ToArabic := roman.ArabicDecoder(handles, fflag.NCHW(), dataflag.Float(), convflag.Mode.CrossCorrelation(), true, 10, int32(metabatchsize*10), learningrates, codingvector, numofneurons, l1regularization, l2regularization)
	utils.CheckError(Encoder.DynamicHidden())
	ToRoman := roman.RomanDecoder(handles, fflag.NCHW(), dataflag.Float(), convflag.Mode.CrossCorrelation(), true, 10, int32(metabatchsize*10), learningrates, codingvector, numofneurons, l1regularization, l2regularization)
	utils.CheckError(Encoder.DynamicHidden())
	romanoutput := putintogpumemRoman(batchedoutput, []int32{10, 1, 28, 28}, fflag.NCHW(), dataflag.Float(), true)
	//Load the batches into gpu mem this is basically the Arabic numbers are place in arabicoutput.T() and arabicnums.DeltaT()
	arabicoutput, arabicnums := putintogpumemArabic(batchesofinputbatches, fflag.NCHW(), dataflag.Float(), []int32{10, 1, 28, 28}, true)
	imager, err := imaging.MakeImager(handles)
	utils.CheckError(err)
	epocs := 200
	///snapshotsize := 500
	MSEArabic, err := loss.CreateMSECalculatorGPU(handles, true)
	utils.CheckError(err)
	MSERoman, err := loss.CreateMSECalculatorGPU(handles, true)
	utils.CheckError(err)

	//Need this memory as an inbetween for the Autoencoder and Loss Function so that it can return the errors to the autoencoder
	RomanOutput, err := layers.BuildIO(fflag.NCHW(), dataflag.Float(), []int32{10, 1, 28, 28}, true)
	utils.CheckError(err)
	ArabicOutput, err := layers.BuildIO(fflag.NCHW(), dataflag.Float(), []int32{10, 1, 28, 28}, true)
	utils.CheckError(err)
	chokepoint := make([]*layers.IO, len(arabicnums))
	for i := range chokepoint {
		chokepoint[i], err = layers.BuildIO(fflag.NCHW(), dataflag.Float(), []int32{10, codingvector, 1, 1}, true)
		utils.CheckError(err)
	}

	//Need this to reshape the output of the autoencoder into something the imager can use to make an image.Imageasdfasd
	imagerlayer, err := layers.BuildIO(fflag.NCHW(), dataflag.Float(), []int32{10, 1, 28, 28}, true)
	utils.CheckError(err)
	metabatchcounter := 0
	var startroman bool
	var metabatchbool bool
	var actuallystartromannow bool
	totalrunimage := make([]image.Image, 0)
	ep := float32(epocs)
	ap := float32(len(arabicnums))

	windows := ui.NewWindows(3, "http://localhost", ":8080", "/index")
	LossDataChan := make(chan []ui.LabelFloat, buffersize)
	lossplotlength := 200
	lossplotindex := 0
	ArabicLoss, ArabicLossLabel := ui.NewVSLossHandle("Batch Loss", LossDataChan, false, lossplotlength, "Arabic Loss")
	windows.AddWindow("Arabic Loss", "", "100", "/ALoss/", ArabicLoss)

	go ui.ServerMain(windows)
	for i := 0; i < epocs; i++ {
		giffer := imaging.NewGiffer(0, 1) //giffer stacks a bunch of images and puts them into a gif
		images := make([]image.Image, 0)
		//Making a lossaray to calculate the loss per batch
		epoclossarabic := float32(0)
		epoclossroman := float32(0)

		for j := range arabicnums {
			if metabatchcounter == 0 {
				metabatchbool = true
			} else {
				metabatchbool = false
			}
			if (!startroman || !metabatchbool) && !actuallystartromannow {
				utils.CheckError(Encoder.ForwardProp(handles, nil, arabicnums[j], chokepoint[j]))
				//choketoroman.LoadTValues(chokepoint.T().Memer())
				utils.CheckError(ToArabic.ForwardProp(handles, nil, chokepoint[j], arabicoutput[j]))
				utils.CheckError(ArabicOutput.LoadTValues(arabicoutput[j].T().Memer()))
				utils.CheckError(MSEArabic.ErrorGPU(handles, ArabicOutput, arabicoutput[j]))
				if lossplotindex < lossplotlength {
					ArabicLossLabel[0].Data[lossplotindex] = MSEArabic.Loss()

				} else {
					lossplotindex = 0
					LossDataChan <- ArabicLossLabel
					ArabicLossLabel[0].Data[lossplotindex] = MSEArabic.Loss()

				}

				epoclossarabic += ArabicLossLabel[0].Data[lossplotindex]
				lossplotindex++

				utils.CheckError(ToArabic.BackPropFilterData(handles, nil, chokepoint[j], ArabicOutput))
				utils.CheckError(Encoder.BackPropFilterData(handles, nil, arabicnums[j], chokepoint[j]))
				if metabatchcounter < (metabatchsize) {
					metabatchcounter++
				} else {
					//	stream.Sync()
					batchsize := 10 * metabatchsize
					utils.CheckError(ToArabic.UpdateWeights(handles, batchsize))
					utils.CheckError(Encoder.UpdateWeights(handles, batchsize))
					metabatchcounter = 0

				}

			} else {
				actuallystartromannow = true
				//utils.CheckError(Encoder.ForwardProp(handles, nil, arabicnums[j], chokepoint))
				//	choketoroman.LoadTValues(chokepoint.T().Memer())
				utils.CheckError(ToRoman.ForwardProp(handles, nil, chokepoint[j], romanoutput))
				utils.CheckError(RomanOutput.LoadTValues(romanoutput.T().Memer()))
				utils.CheckError(MSERoman.ErrorGPU(handles, RomanOutput, romanoutput))
				epoclossroman += MSERoman.Loss()
				utils.CheckError(ToRoman.BackPropFilterData(handles, nil, chokepoint[j], RomanOutput))

				if j%int(ap*(float32(i+1)/ep)) == int(ap*(float32(i+1)/ep))-1 {
					imagerlayer.LoadTValues(romanoutput.T().Memer())
					outputimage, err := imager.TileBatches(handles, imagerlayer, 2, 5)
					utils.CheckError(err)
					images = append(images, outputimage)
				}

				if metabatchcounter < (metabatchsize) {
					metabatchcounter++

				} else {
					batchsize := 10 * metabatchsize
					utils.CheckError(ToRoman.UpdateWeights(handles, batchsize))
					metabatchcounter = 0
				}

			}

		}
		if actuallystartromannow == true {
			somenewimages := make([]image.Image, len(images))
			for j := range images {
				somenewimages[j] = resize.Resize(0, 280, images[j], resize.NearestNeighbor)
			}
			totalrunimage = append(totalrunimage, somenewimages...)
		}

		epoclossroman /= float32(len(arabicnums))
		epoclossarabic /= float32(len(arabicnums))
		stream.Sync()
		fmt.Println("Epoc: ", i, "ROMAN Loss : ", epoclossroman, "ARABIC Loss: ", epoclossarabic)
		if epoclossarabic <= 5 {
			startroman = true
		}

		if (actuallystartromannow == true && epoclossroman <= 3) || i >= epocs-1 {
			fmt.Println("HIT  Loss")
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
func putintogpumemArabic(arabic [][]float32, frmt cudnn.TensorFormat, dtype cudnn.DataType, dimsarabic []int32, memmanaged bool) (output, runs []*layers.IO) {
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
func putintogpumemRoman(romans []float32, dimsroman []int32, frmt cudnn.TensorFormat, dtype cudnn.DataType, memmanaged bool) (output *layers.IO) {

	output, err := layers.BuildNetworkOutputIOFromSlice(frmt, dtype, dimsroman, memmanaged, romans)
	utils.CheckError(err)
	return output
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
