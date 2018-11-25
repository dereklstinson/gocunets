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
	"github.com/dereklstinson/GoCuNets/utils/imaging"
	"github.com/dereklstinson/GoCudnn"
)

const learningrates = .001
const codingvector = int32(10)
const numofneurons = int32(35)
const l1regularization = float32(.0001)
const l2regularization = float32(.0001)

const metabatchsize = 1
const batchsize = 10

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

	//metabatchcounter := 0
	//	metabatchcounter1 := 0
	//	var startroman bool
	//var metabatchbool bool
	//var actuallystartromannow bool

	windows := ui.NewWindows(4, "http://localhost", ":8080", "/index")
	LossDataChan := make(chan []ui.LabelFloat, 2)
	lossplotlength := 1

	imagebuffer := 5
	imagechans := make([]chan image.Image, 3)
	bufferindex := make([]chan int, 3)
	imagerlayer := make([]*layers.IO, 3)
	imagehandlers := make([]*ui.ImageHandler, 3)
	imager := make([]*imaging.Imager, 3)

	for i := range imagehandlers {
		imager[i], err = imaging.MakeImager(handles)
		utils.CheckError(err)
		imagechans[i] = make(chan image.Image, imagebuffer)
		bufferindex[i] = make(chan int, 3)
		imagehandlers[i] = ui.MakeImageHandler(imagebuffer, imagechans[i], bufferindex[i])
		imagerlayer[i], err = layers.BuildIO(fflag.NCHW(), dataflag.Float(), []int32{10, 1, 28, 28}, true)
		utils.CheckError(err)
	}
	refresh := "2000"
	ceiling := float32(10)
	ArabicLoss, EpocLosses := ui.NewVSLossHandle("Epoc Loss", ceiling, LossDataChan, true, lossplotlength, "Arabic Loss", "Roman Loss")
	windows.AddWindow("Arabic Loss Vs Roman Loss", "", refresh, "/ALoss/", ArabicLoss)
	windows.AddWindow("Roman Output", "These are the Roman outputs", refresh, "/romanout/", imagehandlers[0])
	windows.AddWindow("Arabic Output", "These are the Arabic outputs", refresh, "/arabicoutput/", imagehandlers[1])
	windows.AddWindow("Input Image", "Here is the input image for the outputs", refresh, "/arabicinput/", imagehandlers[2])
	go ui.ServerMain(windows)
	EpocLosses[0].Data[0] = ceiling
	EpocLosses[1].Data[0] = ceiling
	LossDataChan <- EpocLosses
	for i := 0; i < epocs; i++ {

		//Making a lossaray to calculate the loss per batch
		epoclossarabic := float32(0)
		epoclossroman := float32(0)

		for j := range arabicnums {

			utils.CheckError(Encoder.ForwardProp(handles, nil, arabicnums[j], chokepoint[j]))

			utils.CheckError(ToArabic.ForwardProp(handles, nil, chokepoint[j], arabicoutput[j]))
			utils.CheckError(ArabicOutput.LoadTValues(arabicoutput[j].T().Memer()))
			utils.CheckError(MSEArabic.ErrorGPU(handles, ArabicOutput, arabicoutput[j]))

			epoclossarabic += MSEArabic.Loss()

			utils.CheckError(ToArabic.BackPropFilterData(handles, nil, chokepoint[j], ArabicOutput))
			utils.CheckError(Encoder.BackPropFilterData(handles, nil, arabicnums[j], chokepoint[j]))

			utils.CheckError(ToRoman.ForwardProp(handles, nil, chokepoint[j], romanoutput))
			utils.CheckError(RomanOutput.LoadTValues(romanoutput.T().Memer()))
			utils.CheckError(MSERoman.ErrorGPU(handles, RomanOutput, romanoutput))
			epoclossroman += MSERoman.Loss()
			utils.CheckError(ToRoman.BackPropFilterData(handles, nil, chokepoint[j], RomanOutput))

			utils.CheckError(ToArabic.UpdateWeights(handles, batchsize))
			utils.CheckError(Encoder.UpdateWeights(handles, batchsize))
			utils.CheckError(ToRoman.UpdateWeights(handles, batchsize))

			for k := range bufferindex {

				var w int
				select {
				case w = <-bufferindex[k]:
				default:
					w = 0
				}

				if w < imagebuffer {
					switch k {
					case 0:
						imagerlayer[k].LoadTValues(romanoutput.T().Memer())
						outputimage, err := imager[k].TileBatches(handles, imagerlayer[k], 2, 5)
						utils.CheckError(err)
						imagechans[k] <- outputimage
					case 1:
						imagerlayer[k].LoadTValues(ArabicOutput.T().Memer())
						outputimage, err := imager[k].TileBatches(handles, imagerlayer[k], 2, 5)
						utils.CheckError(err)
						imagechans[k] <- outputimage
					case 2:
						imagerlayer[k].LoadTValues(arabicnums[j].T().Memer())
						outputimage, err := imager[k].TileBatches(handles, imagerlayer[k], 2, 5)
						utils.CheckError(err)
						imagechans[k] <- outputimage
					}

				} else {

				}
			}

		}

		epoclossroman /= float32(len(arabicnums))
		epoclossarabic /= float32(len(arabicnums))
		EpocLosses[0].Data[0] = epoclossarabic
		EpocLosses[1].Data[0] = epoclossroman
		LossDataChan <- EpocLosses

		fmt.Println("Epoc: ", i, "ROMAN Loss : ", epoclossroman, "ARABIC Loss: ", epoclossarabic)
		err = gocudnn.Cuda{}.CtxSynchronize()
		if err != nil {
			panic(err)
		}
		shuffle(arabicnums, arabicoutput)

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
func shuffle(runs, output []*layers.IO) {
	rand.Shuffle(len(runs), func(i, j int) {
		runs[i], runs[j] = runs[j], runs[i]
		output[i], output[j] = output[j], output[i]
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
