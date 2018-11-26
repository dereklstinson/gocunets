package main

import (
	"fmt"
	"image"
	"math/rand"
	"strconv"

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

	//Need this to reshape the output of the autoencoder into something the imager can use to make an image.Imageasdfasd

	//metabatchcounter := 0
	//	metabatchcounter1 := 0
	//	var startroman bool
	//var metabatchbool bool
	//var actuallystartromannow bool

	windows := ui.NewWindows([]int{4, 3}, "http://localhost", ":8080", "/index")
	LossDataChan := make(chan []ui.LabelFloat, 2)
	lossplotlength := 1

	imagebuffer := 4
	imagechans := make([]chan image.Image, 3)
	bufferindex := make([]chan int, 3)
	parabufferindex := make([]chan int, 4)
	imagerlayer := make([]*layers.IO, 3)
	parachans := make([]chan string, 4)
	imagehandlers := make([]*ui.ImageHandler, 3)
	parahandlers := make([]*ui.ParagraphHandler, 4)
	imager := make([]*imaging.Imager, 3)

	for i := range imagehandlers {
		imager[i], err = imaging.MakeImager(handles)
		utils.CheckError(err)
		imagechans[i] = make(chan image.Image, imagebuffer)
		parachans[i] = make(chan string, imagebuffer)
		bufferindex[i] = make(chan int, 3)
		parabufferindex[i] = make(chan int, 3)
		parahandlers[i] = ui.MakeParagraphHandler(imagebuffer, parachans[i], parabufferindex[i])
		imagehandlers[i] = ui.MakeImageHandler(imagebuffer, imagechans[i], bufferindex[i])
		imagerlayer[i], err = layers.BuildIO(fflag.NCHW(), dataflag.Float(), []int32{10, 1, 28, 28}, true)
		utils.CheckError(err)
	}

	parachans[len(imagehandlers)] = make(chan string, imagebuffer)
	parabufferindex[len(imagehandlers)] = make(chan int, 3)
	parahandlers[len(imagehandlers)] = ui.MakeParagraphHandler(imagebuffer, parachans[len(imagehandlers)], parabufferindex[len(imagehandlers)])
	refresh := "2000" //asdf
	ceiling := float32(10)
	ArabicLoss, EpocLosses := ui.NewVSLossHandle("Epoc Loss", ceiling, LossDataChan, true, lossplotlength, "Arabic Loss", "Roman Loss")
	windows.AddWindow("Arabic Loss Vs Roman Loss", refresh, "/ALoss/", ArabicLoss, "/ARLoss/", parahandlers[3])
	windows.AddWindow("Roman Output", refresh, "/romanout/", imagehandlers[0], "/romanp/", parahandlers[0])
	windows.AddWindow("Arabic Output", refresh, "/arabicoutput/", imagehandlers[1], "/arabp/", parahandlers[1])
	windows.AddWindow("Input Image", refresh, "/arabicinput/", imagehandlers[2], "/arabinput/", parahandlers[2])

	encoderhidden := Encoder.HiddenIOandWeightCount()

	encoderimagechans := make([]chan image.Image, encoderhidden)
	encoderparachans := make([]chan string, encoderhidden)

	encoderbufferindex := make([]chan int, encoderhidden)
	encoderparabufferindex := make([]chan int, encoderhidden)

	encoderimagehandlers := make([]*ui.ImageHandler, encoderhidden)
	encoderparahandlers := make([]*ui.ParagraphHandler, encoderhidden)

	for i := 0; i < encoderhidden; i++ {
		encoderimagechans[i] = make(chan image.Image, imagebuffer)
		encoderparachans[i] = make(chan string, imagebuffer)
		encoderbufferindex[i] = make(chan int, 3)
		encoderparabufferindex[i] = make(chan int, 3)
		encoderimagehandlers[i] = ui.MakeImageHandler(imagebuffer, encoderimagechans[i], encoderbufferindex[i])
		encoderparahandlers[i] = ui.MakeParagraphHandler(imagebuffer, encoderparachans[i], encoderparabufferindex[i])
		windows.AddWindow("Hidden "+strconv.Itoa(i), refresh, "/encode"+strconv.Itoa(i)+"/", encoderimagehandlers[i], "/pencode"+strconv.Itoa(i)+"/", encoderparahandlers[i])
	}
	dencoderimagechans := make([]chan image.Image, encoderhidden)
	dencoderparachans := make([]chan string, encoderhidden)

	dencoderbufferindex := make([]chan int, encoderhidden)
	dencoderparabufferindex := make([]chan int, encoderhidden)

	dencoderimagehandlers := make([]*ui.ImageHandler, encoderhidden)
	dencoderparahandlers := make([]*ui.ParagraphHandler, encoderhidden)
	for i := 0; i < encoderhidden; i++ {
		dencoderimagechans[i] = make(chan image.Image, imagebuffer)
		dencoderparachans[i] = make(chan string, imagebuffer)
		dencoderbufferindex[i] = make(chan int, 3)
		dencoderparabufferindex[i] = make(chan int, 3)
		dencoderimagehandlers[i] = ui.MakeImageHandler(imagebuffer, dencoderimagechans[i], dencoderbufferindex[i])
		dencoderparahandlers[i] = ui.MakeParagraphHandler(imagebuffer, dencoderparachans[i], dencoderparabufferindex[i])
		windows.AddWindow("Hidden "+strconv.Itoa(i), refresh, "/dencode"+strconv.Itoa(i)+"/", encoderimagehandlers[i], "/dpencode"+strconv.Itoa(i)+"/", encoderparahandlers[i])
	}
	/*
		//arabic
		decoderarabic := ToArabic.HiddenIOandWeightCount()
		arabicimagechans := make([]chan image.Image, decoderarabic)
		arabicbufferindex := make([]chan int, decoderarabic)
		arabicparabufferindex := make([]chan int, decoderarabic)
		arabicimagerlayer := make([]*layers.IO, decoderarabic)
		arabicparachans := make([]chan string, decoderarabic)
		arabicimagehandlers := make([]*ui.ImageHandler, decoderarabic)
		arabicparahandlers := make([]*ui.ParagraphHandler, decoderarabic)
		arabicimager := make([]*imaging.Imager, decoderarabic)
		//roman
		decoderroman := ToRoman.HiddenIOandWeightCount()
		romanimagechans := make([]chan image.Image, decoderroman)
		romanbufferindex := make([]chan int, decoderroman)
		romanparabufferindex := make([]chan int, decoderroman)
		romanimagerlayer := make([]*layers.IO, decoderroman)
		romanparachans := make([]chan string, decoderroman)
		romanimagehandlers := make([]*ui.ImageHandler, decoderroman)
		romanparahandlers := make([]*ui.ParagraphHandler, decoderroman)
		romanimager := make([]*imaging.Imager, 3)
	*/
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
				romanloss := epoclossroman
				arabicloss := epoclossarabic
				romanloss /= float32(j)
				arabicloss /= float32(j)
				for k := range parabufferindex {

					var w int
					select {
					case w = <-parabufferindex[k]:
					default:
						w = 0
					}

					if w < imagebuffer {
						switch k {
						case 0:
							//imagerlayer[k].LoadTValues(romanoutput.T().Memer())
							//outputimage, err := imager[k].TileBatches(handles, imagerlayer[k], 2, 5)
							//utils.CheckError(err)

							parachans[k] <- fmt.Sprintf("Roman Loss:%-0.2f", romanloss)
						case 1:
							//	imagerlayer[k].LoadTValues(ArabicOutput.T().Memer())
							//	outputimage, err := imager[k].TileBatches(handles, imagerlayer[k], 2, 5)
							//	utils.CheckError(err)

							parachans[k] <- fmt.Sprintf("Arabic Loss: %-0.2f", arabicloss)
						case 2:
							//	imagerlayer[k].LoadTValues(arabicnums[j].T().Memer())
							//	outputimage, err := imager[k].TileBatches(handles, imagerlayer[k], 2, 5)
							//	utils.CheckError(err)
							parachans[k] <- fmt.Sprintf("Epoc: %d, Batch %d", i, j)
						case 3:
							//	imagerlayer[k].LoadTValues(arabicnums[j].T().Memer())
							//	outputimage, err := imager[k].TileBatches(handles, imagerlayer[k], 2, 5)
							//	utils.CheckError(err)
							parachans[k] <- fmt.Sprintf("Epoc: %d, Batch %d, Roman Loss: %-0.2f, Arabic Loss: %-0.2f", i, j, romanloss, arabicloss)
						}

					} else {

					}
				}

			}
			Eminmax, err := Encoder.GetMinMaxes(handles)
			utils.CheckError(err)
			Eimages, err := Encoder.GetLayerImages(handles, 0, 0)
			utils.CheckError(err)

			for k := range encoderbufferindex {

				var w int
				select {
				case w = <-encoderbufferindex[k]:
				default:
					w = 0
				}

				if w < imagebuffer {
					x := Eimages[k].X
					filing.WriteImage("/home/derek/Desktop/Weightimage/", "THISFILE"+strconv.Itoa(i)+"0000"+strconv.Itoa(j)+"OOOO"+strconv.Itoa(k), x)
					encoderimagechans[k] <- x
				} else {

				}

				select {
				case w = <-encoderparabufferindex[k]:
				default:
					w = 0
				}
				if w < imagebuffer {
					name := Eminmax[k].Name
					if name == "CNN-Transpose" || name == "CNN" {
						wminx := Eminmax[k].Weights.Minx
						wmaxx := Eminmax[k].Weights.Maxx
						bminx := Eminmax[k].Bias.Minx
						bmaxx := Eminmax[k].Weights.Maxx
						encoderparachans[k] <- fmt.Sprintf("Layer: %s %d, Weights:{Min: %f ,Max: %f} Bias{Min: %f ,Max: %f}", name, k, wminx, wmaxx, bminx, bmaxx)
					} else {
						wminx := Eminmax[k].Weights.Minx
						wmaxx := Eminmax[k].Weights.Maxx
						encoderparachans[k] <- fmt.Sprintf("Layer: %s %d, Weights:{Min: %f ,Max: %f}", name, k, wminx, wmaxx)
					}

				}
				select {
				case w = <-dencoderbufferindex[k]:
				default:
					w = 0
				}

				if w < imagebuffer {
					dencoderimagechans[k] <- Eimages[k].DX
				} else {

				}
				select {
				case w = <-dencoderparabufferindex[k]:
				default:
					w = 0
				}
				if w < imagebuffer {
					name := Eminmax[k].Name
					if name == "CNN-Transpose" || name == "CNN" {
						wminx := Eminmax[k].Weights.Mindx
						wmaxx := Eminmax[k].Weights.Maxdx
						bminx := Eminmax[k].Bias.Mindx
						bmaxx := Eminmax[k].Weights.Maxdx
						dencoderparachans[k] <- fmt.Sprintf("Layer: %s %d, Weights:{Min: %f ,Max: %f} Bias{Min: %f ,Max: %f}", name, k, wminx, wmaxx, bminx, bmaxx)
					} else {
						wminx := Eminmax[k].Weights.Mindx
						wmaxx := Eminmax[k].Weights.Maxdx
						dencoderparachans[k] <- fmt.Sprintf("Layer: %s %d, Weights:{Min: %f ,Max: %f}", name, k, wminx, wmaxx)
					}

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
		/*
			Eminmax, err := Encoder.GetMinMaxes(handles)
			utils.CheckError(err)
			Eimages, err := Encoder.GetLayerImages(handles, 10, 10)
			utils.CheckError(err)

				for k := range encoderbufferindex {

					var w int
					select {
					case w = <-encoderbufferindex[k]:
					default:
						w = 0
					}

					if w < imagebuffer {
						encoderimagechans[k] <- Eimages[k].X
					} else {

					}

					select {
					case w = <-encoderparabufferindex[k]:
					default:
						w = 0
					}
					if w < imagebuffer {
						name := Eminmax[k].Name
						if name == "CNN-Transpose" || name == "CNN" {
							wminx := Eminmax[k].Weights.Minx
							wmaxx := Eminmax[k].Weights.Maxx
							bminx := Eminmax[k].Bias.Minx
							bmaxx := Eminmax[k].Weights.Maxx
							encoderparachans[k] <- fmt.Sprintf("Layer: %s %d, Weights:{Min: %f ,Max: %f} Bias{Min: %f ,Max: %f}", name, k, wminx, wmaxx, bminx, bmaxx)
						} else {
							wminx := Eminmax[k].Weights.Minx
							wmaxx := Eminmax[k].Weights.Maxx
							encoderparachans[k] <- fmt.Sprintf("Layer: %s %d, Weights:{Min: %f ,Max: %f}", name, k, wminx, wmaxx)
						}

					}
					select {
					case w = <-dencoderbufferindex[k]:
					default:
						w = 0
					}

					if w < imagebuffer {
						dencoderimagechans[k] <- Eimages[k].DX
					} else {

					}
					select {
					case w = <-dencoderparabufferindex[k]:
					default:
						w = 0
					}
					if w < imagebuffer {
						name := Eminmax[k].Name
						if name == "CNN-Transpose" || name == "CNN" {
							wminx := Eminmax[k].Weights.Mindx
							wmaxx := Eminmax[k].Weights.Maxdx
							bminx := Eminmax[k].Bias.Mindx
							bmaxx := Eminmax[k].Weights.Maxdx
							dencoderparachans[k] <- fmt.Sprintf("Layer: %s %d, Weights:{Min: %f ,Max: %f} Bias{Min: %f ,Max: %f}", name, k, wminx, wmaxx, bminx, bmaxx)
						} else {
							wminx := Eminmax[k].Weights.Mindx
							wmaxx := Eminmax[k].Weights.Maxdx
							dencoderparachans[k] <- fmt.Sprintf("Layer: %s %d, Weights:{Min: %f ,Max: %f}", name, k, wminx, wmaxx)
						}

					}
				}
		*/
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
