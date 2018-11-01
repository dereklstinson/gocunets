package main

import (
	"fmt"
	"math/rand"
	"strconv"

	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/utils"
	"github.com/dereklstinson/GoCuNets/utils/filing"
	"github.com/dereklstinson/GoCuNets/utils/imaging"

	"github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCudnn"

	"github.com/dereklstinson/GoCuNets/testing/mnist/dfuncs"
	"github.com/dereklstinson/GoCuNets/testing/mnisttoroman/roman"
)

func main() {
	//const romanimagelocation= "/home/derek/go/src/github.com/dereklstinson/testing/mnist/roman/"
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
	handles := gocunets.CreateHandle(devs[0])
	stream, err := gocudnn.Cuda{}.CreateBlockingStream()
	utils.CheckError(err)
	handles.SetStream(stream)
	var dataflag gocudnn.DataTypeFlag
	var convflag gocudnn.ConvolutionFlags
	var fflag gocudnn.TensorFormatFlag
	toroman := roman.Encoder(handles, fflag.NCHW(), dataflag.Float(), convflag.Mode.CrossCorrelation(), true, 10)
	utils.CheckError(toroman.DynamicHidden())
	romanoutput, arabicnums := putintogpumem(batchedoutput, []int32{10, 1, 28, 28}, batchesofinputbatches, fflag.NCHW(), dataflag.Float(), []int32{10, 1, 28, 28}, true)
	imager, err := imaging.MakeImager(handles.XHandle())
	utils.CheckError(err)
	epocs := 100
	for i := 0; i < epocs; i++ {
		for j := range arabicnums {
			utils.CheckError(toroman.ForwardProp(handles, nil, arabicnums[j], romanoutput))
			stream.Sync()
			utils.CheckError(toroman.BackPropFilterData(handles, nil, arabicnums[j], romanoutput))
			stream.Sync()
			utils.CheckError(toroman.UpdateWeights(handles, 10))
			stream.Sync()
		}
		shuffle(arabicnums)
		outputimage, err := imager.TileBatches(handles.XHandle(), romanoutput, 2, 5)
		utils.CheckError(err)
		stream.Sync()

		utils.CheckError(filing.WriteImage(imagesave, "RomanEpoc"+strconv.Itoa(i), outputimage))
	}
}

func putintogpumem(romans []float32, dimsroman []int32, arabic [][]float32, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dimsarabic []int32, memmanaged bool) (output *layers.IO, runs []*layers.IO) {
	var err error
	runs = make([]*layers.IO, 0)
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
