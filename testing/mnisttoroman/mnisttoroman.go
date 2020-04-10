package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"

	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCudnn/gocu"

	"github.com/dereklstinson/GoCuNets/testing/mnist/dfuncs"
	"github.com/dereklstinson/GoCuNets/testing/mnisttoroman/roman"
	"github.com/dereklstinson/GoCuNets/ui"
	"github.com/dereklstinson/GoCuNets/utils"
	"github.com/dereklstinson/GoCuNets/utils/imaging"
)

const ipAddress = "http://192.168.55.11" //"http://localhost"
const learningrates = .005
const codingvector = int32(12)
const numofneurons = int32(128)
const l1regularization = float32(0.000001)
const l2regularization = float32(0.0001)

const metabatchsize = 1
const batchsize = 10
const vsbatchskip = 4

func main() {
	runtime.LockOSThread()
	rand.Seed(time.Now().UnixNano())
	//gocunets.PerformanceDebugging()
	const romanimagelocation = "../mnist/roman/"
	const filedirectory = "../mnist/files/"
	const mnistfilelabel = "train-labels.idx1-ubyte"
	const mnistimage = "train-images.idx3-ubyte"
	const mnistfilelabeltest = "test-labels.idx1-ubyte"
	const mnistimagetest = "test-images.idx3-ubyte"
	const imagesave = "/home/derek/Desktop/RomanOutput/"
	romannums := roman.GetRoman(romanimagelocation)
	mnistdata, err := dfuncs.LoadMNIST(filedirectory, mnistfilelabel, mnistimage)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(romannums))
	fmt.Println(len(mnistdata))
	avg := dfuncs.FindAverage(mnistdata)
	mnistpixelsoft := dfuncs.MakeEncodeeSoftmaxPerPixelCopy(mnistdata)
	mnistdata = dfuncs.NormalizeData(mnistdata, avg)

	romannums = roman.EncodeSoftmaxPerPixel(romannums)
	sectioned := makenumbers(romannums, mnistdata, mnistpixelsoft)
	batchedoutput := makeoutputbatch(sectioned)
	batchesofinputbatches, batchesofoutputbatches := makemnistbatches(sectioned, 28*28, 28*28*2)
	fmt.Println(len(batchedoutput)) //batch is 10 chanel is 1 and dims are 28 by 28
	fmt.Println(len(batchesofinputbatches))
	//runsize := len(batchesofinputbatches)
	devs, err := gocunets.GetDeviceList()
	utils.CheckError(err)
	worker := gocu.NewWorker(devs[0])
	handles := gocunets.CreateHandle(worker, devs[0], rand.Uint64())
	stream, err := gocunets.CreateStream()

	utils.CheckError(err)
	utils.CheckError(handles.SetStream(stream))

	builder := gocunets.CreateBuilder(handles)
	builder.Dtype.Float()
	builder.Frmt.NCHW()
	builder.AMode.Leaky()
	builder.Nan.NotPropigate()
	builder.Cmode.Convolution()
	intputdimss := []int32{10, 1, 28, 28}
	hiddenoutputchannels := []int32{8, 8, 8}

	ArabicInputTensor, err := builder.CreateTensor(intputdimss)
	decoderoutputchannels := int32(8)
	utils.CheckError(err)
	var outputchannels = int32(2)
	fmt.Println("Making Encoder")
	Encoder := roman.ArabicEncoder(builder, batchsize, decoderoutputchannels, hiddenoutputchannels, learningrates, l1regularization, l2regularization, ArabicInputTensor)
	fmt.Println("Making ToArabic")
	EncoderOutputY := Encoder.GetTensorY()
	EncoderOutputDY := Encoder.GetTensorDY()
	ReverseConcat, err := gocunets.CreateReverseConcat(handles)
	if err != nil {
		panic(err)
	}
	splitdims, err := ReverseConcat.FindOutputDims(EncoderOutputY, 2)
	if err != nil {
		panic(err)
	}
	fmt.Println(splitdims[0], splitdims[1])
	ReverseConcat.SetInputSource(EncoderOutputY)
	ReverseConcat.SetInputDeltaSource(EncoderOutputDY)
	ArabicX, err := builder.CreateTensor(splitdims[0])
	if err != nil {
		panic(err)
	}
	ArabicDX, err := builder.CreateTensor(splitdims[0])
	if err != nil {
		panic(err)
	}
	RomanX, err := builder.CreateTensor(splitdims[1])
	if err != nil {
		panic(err)
	}
	RomanDX, err := builder.CreateTensor(splitdims[1])
	if err != nil {
		panic(err)
	}
	ReverseConcat.SetOutputDests([]*gocunets.Tensor{ArabicX, RomanX})
	ReverseConcat.SetOutputDeltaDests([]*gocunets.Tensor{ArabicDX, RomanDX})
	ToArabic := roman.ArabicDecoder(builder, batchsize, outputchannels, hiddenoutputchannels, learningrates, l1regularization, l2regularization, ArabicX, ArabicDX)

	ToRoman := roman.RomanDecoder(builder, batchsize, outputchannels, hiddenoutputchannels, learningrates, l1regularization, l2regularization, RomanX, RomanDX)
	fmt.Println("Done Making Networks")
	fmt.Println("Put Roman Images into gpu")

	romanoutput := putintogpumemRoman(handles, batchedoutput, []int32{10, 2, 28, 28}, builder)
	//fmt.Println(romanoutput)
	//Load the batches into gpu mem this is basically the Arabic numbers are place in arabicoutput.T() and arabicnums.DeltaT()
	fmt.Println("Put Arabic Images into GPU")
	arabicoutput, arabicnums := putintogpumemArabic(handles, builder, batchesofinputbatches, []int32{10, 1, 28, 28}, batchesofoutputbatches, []int32{10, 2, 28, 28})
	fmt.Println("Done putting images into GPU")
	epocs := 200
	///snapshotsize := 500

	//Need this memory as an inbetween for the Autoencoder and Loss Function so that it can return the errors to the autoencoder

	windows := ui.NewWindows(ipAddress, ":8080", "/index")

	LossDataChan := make(chan []ui.LabelFloat, 2)
	//lossplotlength := 100

	imagebuffer := 4
	imagehandlers := make([]*ui.ImageHandlerV2, 3)

	imagerlayer := make([]*gocunets.Tensor, 3)

	imager := make([]*imaging.Imager, 3)
	inputimagesize := []int32{10, 1, 28, 28}
	outputimagesize := []int32{10, 2, 28, 28}
	imagesizes := [][]int32{inputimagesize, outputimagesize, outputimagesize}
	for i := range imager {
		imager[i], err = imaging.MakeImager(handles.Handler)
		utils.CheckError(err)
		//	parahandlers[i] = ui.MakeParagraphHandlerV2(imagebuffer)
		imagehandlers[i] = ui.MakeImageHandlerV2(imagebuffer, "/mnt/share/datasets/uiimage/GoCu.png")
		imagerlayer[i], err = builder.CreateTensor(imagesizes[i])
		utils.CheckError(err)
	}

	refresh := "2000"
	ceiling := float32(15)
	ArabicLoss, EpocLosses := ui.NewVSLossHandle("Epoc Loss", ceiling, LossDataChan, false, len(arabicnums), 1, "Arabic Loss", "Roman Loss")
	windows.AddNetIO("Arabic Loss Vs Roman Loss", "500", "/ALoss/", ArabicLoss, "", nil, 4, true, false)
	windows.AddNetIO("Input Image", refresh, "/arabicinput/", imagehandlers[0], "", nil, 4, false, false)
	windows.AddNetIO("Arabic Output Current", refresh, "/arabicoutputCurrent/", imagehandlers[1], "", nil, 4, false, false)
	windows.AddNetIO("Roman Output Current", refresh, "/romanoutputCurrent/", imagehandlers[2], "", nil, 4, false, true)
	/*networkstats := make([]*ui.ParagraphHandlerV2, 3)
	for i := range networkstats {
		networkstats[i] = ui.MakeParagraphHandlerV2(imagebuffer)
	}
	*/
	go ui.ServerMain(windows)

	//LossDataChan <- EpocLosses
	//	outsidecounter := 0
	var updatecounter int
	for i := 0; i < epocs; i++ {

		//Making a lossaray to calculate the loss per batch
		epoclossarabic := float32(0)
		epoclossroman := float32(0)

		for j := range arabicnums {
			if updatecounter > 100 {
				ToRoman.SetTensorDX(RomanDX)
			}
			utils.CheckError(ArabicInputTensor.LoadMem(handles.Handler, arabicnums[j], arabicnums[j].SIB()))
			utils.CheckError(ToRoman.GetTensorDY().LoadMem(handles.Handler, romanoutput, romanoutput.SIB()))
			utils.CheckError(ToArabic.GetTensorDY().LoadMem(handles.Handler, arabicoutput[j], arabicoutput[j].SIB()))
			//utils.CheckError(ToRoman.GetTensorDY().LoadMem(handles.Handler, arabicoutput[j], arabicoutput[j].SIB()))
			//Encode Image
			utils.CheckError(Encoder.Forward())
			utils.CheckError(stream.Sync())
			//Split output from encoder
			utils.CheckError(ReverseConcat.Forward())
			utils.CheckError(stream.Sync())
			//Decode Splitted Outputs
			//Encoder.GetTensorY().TogglePrintValueForStringer()
			//fmt.Println(Encoder.GetTensorY())
			//Encoder.GetTensorY().TogglePrintValueForStringer()
			//ToArabic.GetTensorX().TogglePrintValueForStringer()
			//fmt.Println(ToArabic.GetTensorX())
			//ToArabic.GetTensorX().TogglePrintValueForStringer()
			//for {
			//
			//}
			utils.CheckError(ToArabic.Forward())
			utils.CheckError(stream.Sync())
			utils.CheckError(ToRoman.Forward())
			utils.CheckError(stream.Sync())
			//Send the errors Backward
			utils.CheckError(ToArabic.Backward())
			utils.CheckError(stream.Sync())
			utils.CheckError(ToRoman.Backward())
			utils.CheckError(stream.Sync())
			utils.CheckError(ReverseConcat.Backward())
			utils.CheckError(stream.Sync())
			utils.CheckError(Encoder.Backward())
			utils.CheckError(stream.Sync())
			utils.CheckError(ToArabic.Update(updatecounter))
			utils.CheckError(Encoder.Update(updatecounter))
			utils.CheckError(ToRoman.Update(updatecounter))
			updatecounter++
			utils.CheckError(stream.Sync())
			outputs := []*gocunets.Tensor{arabicnums[j], ToArabic.GetTensorY(), ToRoman.GetTensorY()}
			lossroman := ToRoman.Classifier.GetAverageBatchLoss()
			lossarabic := ToArabic.Classifier.GetAverageBatchLoss()
			//	fmt.Println(lossroman, lossarabic)
			if math.IsNaN(float64(lossarabic)) {
				if math.IsNaN(float64(lossroman)) {
					panic("lossarabic and lossroman is nan")
				}
				panic("lossarabic is nan")
			}
			if math.IsNaN(float64(lossroman)) {
				if math.IsNaN(float64(lossarabic)) {
					panic("lossarabic and lossroman is nan")
				}
				panic("lossroman is nan")
			}
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
					imagerlayer[k].LoadMem(handles.Handler, outputs[k], outputs[k].SIB())
					utils.CheckError(handles.Sync())
					outputimage, err := imager[k].TileBatches(handles.Handler, imagerlayer[k].Tensor, 2, 5, 28, 28)
					utils.CheckError(err)
					utils.CheckError(handles.Sync())
					imagehandlers[k].Image(outputimage)

				}
			}

		}
		/*
			for j := range networkstats {
				var w int
				select {
				case w = <-networkstats[j].Buffer():
				default:
					w = 0
				}
				if w < imagebuffer {
					paras, err := networks[j].GetHTMLedStats(handles.Handler)
					utils.CheckError(err)
					networkstats[j].Paragraph(paras)
				}

			}
		*/
		epoclossarabic /= float32(len(arabicnums))
		epoclossroman /= float32(len(arabicnums))

		fmt.Println("Epoc: ", i, "ROMAN Loss : ", epoclossroman, "ARABIC Loss: ", epoclossarabic)

		shuffle(arabicnums, arabicoutput)
	}

}

func putintogpumemArabic(handles *gocunets.Handle, builder *gocunets.Builder,
	inputarabic [][]float32, inputdims []int32, outputarabic [][]float32, outputdims []int32) (targetoutput, input []*gocunets.Tensor) {
	var err error
	input = make([]*gocunets.Tensor, len(inputarabic))
	targetoutput = make([]*gocunets.Tensor, len(outputarabic))
	for i := range input {
		//fmt.Println("addtensor", i)
		input[i], err = builder.CreateTensor(inputdims)
		utils.CheckError(err)
		//ptr, err := gocudnn.MakeGoPointer(arabic[i])
		//utils.CheckError(err)
		utils.CheckError(input[i].LoadValuesFromSLice(handles.Handler, inputarabic[i], int32(len(inputarabic[i]))))
		targetoutput[i], err = builder.CreateTensor(outputdims)
		utils.CheckError(err)
		utils.CheckError(targetoutput[i].LoadValuesFromSLice(handles.Handler, outputarabic[i], int32(len(outputarabic[i]))))
	}
	return targetoutput, input
}
func putintogpumemRoman(handles *gocunets.Handle, romans []float32, dimsroman []int32, builder *gocunets.Builder) (targetoutput *gocunets.Tensor) {
	var err error

	targetoutput, err = builder.CreateTensor(dimsroman)
	fmt.Println("Create Tensor roman")
	utils.CheckError(err)
	utils.CheckError(targetoutput.LoadValuesFromSLice(handles.Handler, romans, int32(len(romans))))
	fmt.Println("Done laoding roman slice")
	return targetoutput
}
func shuffle(runs, output []*gocunets.Tensor) {
	rand.Shuffle(len(runs), func(i, j int) {
		runs[i], runs[j] = runs[j], runs[i]
		output[i], output[j] = output[j], output[i]
		//output1[i], output1[j] = output1[j], output1[i]
	})
}

func makemnistbatches(sections []number, inputsize, outputsize int) (input [][]float32, output [][]float32) {
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
	input = make([][]float32, numofbatches)
	output = make([][]float32, numofbatches)
	for i := range input {
		input[i] = make([]float32, numinbatches*inputsize)
		output[i] = make([]float32, numinbatches*outputsize)
	}
	for i := range sections {
		for j := 0; j < numofbatches; j++ {
			for k := range sections[i].mnist[j].Data {

				input[j][i*inputsize+k] = sections[i].mnist[j].Data[k]
			}
			for k := range sections[i].mnistlabel[j].Data {
				output[j][i*outputsize+k] = sections[i].mnistlabel[j].Data[k]
			}

		}

	}
	return input, output
}

func makeoutputbatch(sections []number) []float32 {
	batches := make([]float32, len(sections)*28*28*2)
	for i := range sections {
		for j := range sections[i].roman.Data {
			batches[i*28*28*2+j] = sections[i].roman.Data[j]
		}
	}
	return batches
}

func makenumbers(rmn []roman.Roman, mnist, mnistlabeled []dfuncs.LabeledData) []number {
	sections := make([]number, len(rmn))
	for i := range mnist {
		nmbr := mnist[i].Number
		sections[nmbr].mnist = append(sections[nmbr].mnist, mnist[i])
		sections[nmbr].mnistlabel = append(sections[nmbr].mnistlabel, mnistlabeled[i])
	}
	for i := range rmn {
		nmbr := rmn[i].Number
		sections[nmbr].roman = rmn[i]
	}
	return sections
}

type number struct {
	mnist          []dfuncs.LabeledData
	mnistdims      []int32
	mnistlabel     []dfuncs.LabeledData
	mnistlabeldims []int32
	roman          roman.Roman
}
