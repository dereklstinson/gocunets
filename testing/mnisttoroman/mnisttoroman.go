package main

import (
	"fmt"
	"math/rand"

	gocunets "github.com/dereklstinson/GoCuNets"
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

const ipAddress = "http://localhost" //"http://localhost"
const learningrates = .005
const codingvector = int32(12)
const numofneurons = int32(128)
const l1regularization = float32(0.0)
const l2regularization = float32(0.0)

const metabatchsize = 1
const batchsize = 10
const vsbatchskip = 4

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
	utils.CheckError(ToArabic.DynamicHidden())
	ToRoman := roman.RomanDecoder(handles, fflag.NCHW(), dataflag.Float(), convflag.Mode.CrossCorrelation(), true, 10, int32(metabatchsize*10), learningrates, codingvector, numofneurons, l1regularization, l2regularization)
	utils.CheckError(ToRoman.DynamicHidden())
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

	windows := ui.NewWindows(ipAddress, ":8080", "/index")

	LossDataChan := make(chan []ui.LabelFloat, 2)
	//lossplotlength := 100

	imagebuffer := 4
	imagehandlers := make([]*ui.ImageHandlerV2, 3)

	imagerlayer := make([]*layers.IO, 3)

	imager := make([]*imaging.Imager, 3)

	for i := range imager {
		imager[i], err = imaging.MakeImager(handles)
		utils.CheckError(err)
		//	parahandlers[i] = ui.MakeParagraphHandlerV2(imagebuffer)
		imagehandlers[i] = ui.MakeImageHandlerV2(imagebuffer)
		imagerlayer[i], err = layers.BuildIO(fflag.NCHW(), dataflag.Float(), []int32{10, 1, 28, 28}, true)
		utils.CheckError(err)
	}

	refresh := "2000"
	ceiling := float32(15)
	ArabicLoss, EpocLosses := ui.NewVSLossHandle("Epoc Loss", ceiling, LossDataChan, false, len(arabicnums), 1, "Arabic Loss", "Roman Loss")
	windows.AddNetIO("Arabic Loss Vs Roman Loss", "500", "/ALoss/", ArabicLoss, "", nil, 4, true, false)
	windows.AddNetIO("Input Image", refresh, "/arabicinput/", imagehandlers[0], "", nil, 4, false, false)
	windows.AddNetIO("Arabic Output Current", refresh, "/arabicoutputCurrent/", imagehandlers[1], "", nil, 4, false, false)
	windows.AddNetIO("Roman Output Current", refresh, "/romanoutputCurrent/", imagehandlers[2], "", nil, 4, false, true)
	networkstats := make([]*ui.ParagraphHandlerV2, 3)
	for i := range networkstats {
		networkstats[i] = ui.MakeParagraphHandlerV2(imagebuffer)
	}
	networks := []*gocunets.Network{Encoder, ToArabic, ToRoman}
	windows.AddStats("Arabic Encoder Stats", "5000", "/ArabicEncoder/", networkstats[0], 3, true, false)
	windows.AddStats("Arabic Decoder Stats", "5000", "/ArabicDecoder/", networkstats[1], 3, false, false)
	windows.AddStats("Roman Decoder Stats", "5000", "/RomanDecoder/", networkstats[2], 3, false, true)
	go ui.ServerMain(windows)

	//LossDataChan <- EpocLosses
	//	outsidecounter := 0
	for i := 0; i < epocs; i++ {

		//Making a lossaray to calculate the loss per batch
		epoclossarabic := float32(0)
		epoclossroman := float32(0)

		for j := range arabicnums {
			//	if outsidecounter >= lossplotlength {

			//	outsidecounter = 0

			//	}
			utils.CheckError(Encoder.ForwardProp(handles, nil, arabicnums[j], chokepoint[j]))
			utils.CheckError(stream.Sync())
			utils.CheckError(ToArabic.ForwardProp(handles, nil, chokepoint[j], arabicoutput[j]))
			utils.CheckError(stream.Sync())
			utils.CheckError(ToRoman.ForwardProp(handles, nil, chokepoint[j], romanoutput))
			utils.CheckError(stream.Sync())
			utils.CheckError(RomanOutput.LoadTValues(romanoutput.T().Memer()))
			utils.CheckError(stream.Sync())
			utils.CheckError(MSERoman.ErrorGPU(handles, RomanOutput, romanoutput))
			utils.CheckError(stream.Sync())

			utils.CheckError(ToRoman.BackPropFilterData(handles, nil, chokepoint[j], RomanOutput))
			utils.CheckError(stream.Sync())
			utils.CheckError(ArabicOutput.LoadTValues(arabicoutput[j].T().Memer()))
			utils.CheckError(stream.Sync())
			utils.CheckError(MSEArabic.ErrorGPU(handles, ArabicOutput, arabicoutput[j]))
			utils.CheckError(stream.Sync())
			utils.CheckError(ToArabic.BackPropFilterData(handles, nil, chokepoint[j], ArabicOutput))
			utils.CheckError(stream.Sync())
			utils.CheckError(Encoder.BackPropFilterData(handles, nil, arabicnums[j], chokepoint[j]))
			utils.CheckError(stream.Sync())

			utils.CheckError(ToArabic.UpdateWeights(handles, batchsize))
			utils.CheckError(Encoder.UpdateWeights(handles, batchsize))
			utils.CheckError(ToRoman.UpdateWeights(handles, batchsize))
			utils.CheckError(stream.Sync())

			//	utils.CheckError(ToArabic.UpdateWeights(handles, batchsize))
			//	utils.CheckError(Encoder.UpdateWeights(handles, batchsize))
			//	utils.CheckError(ToRoman.UpdateWeights(handles, batchsize))
			//	utils.CheckError(stream.Sync())
			//	outsidecounter++
			outputs := []*layers.IO{arabicnums[j], ArabicOutput, RomanOutput}
			lossroman := MSERoman.Loss()
			lossarabic := MSEArabic.Loss()
			if lossroman == 0 || lossarabic == 0 {
				fmt.Println("At Batch", j, "Roman Loss:", lossroman, ", Arabic Loss:", lossarabic)
			}
			epoclossroman += lossroman

			epoclossarabic += lossarabic
			//	if j%vsbatchskip == 0 {
			EpocLosses[0].Data = lossarabic
			EpocLosses[1].Data = lossroman
			LossDataChan <- EpocLosses
			//	}

			for k := range imagehandlers {
				var w int
				select {
				case w = <-imagehandlers[k].Buffer():
				default:
					w = 0
				}
				if imagebuffer > w {
					utils.CheckError(handles.Sync())
					imagerlayer[k].LoadTValues(outputs[k].T().Memer())
					outputimage, err := imager[k].TileBatches(handles, imagerlayer[k], 2, 5, 28, 28)
					utils.CheckError(err)
					imagehandlers[k].Image(outputimage)

				}
			}

		}

		for j := range networkstats {
			var w int
			select {
			case w = <-networkstats[j].Buffer():
			default:
				w = 0
			}
			if w < imagebuffer {
				paras, err := networks[j].GetHTMLedStats(handles)
				utils.CheckError(err)
				networkstats[j].Paragraph(paras)
			}

		}

		epoclossarabic /= float32(len(arabicnums))
		epoclossroman /= float32(len(arabicnums))

		fmt.Println("Epoc: ", i, "ROMAN Loss : ", epoclossroman, "ARABIC Loss: ", epoclossarabic)

		shuffle(arabicnums, arabicoutput)
	}

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
