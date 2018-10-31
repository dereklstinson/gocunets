package main

import (
	"fmt"

	"github.com/dereklstinson/GoCuNets/testing/mnist/dfuncs"
	"github.com/dereklstinson/GoCuNets/testing/mnisttoroman/roman"
)

func main() {
	//const romanimagelocation= "/home/derek/go/src/github.com/dereklstinson/testing/mnist/roman/"
	const romanimagelocation = "../mnist/roman/"
	const filedirectory = "../mnist/files/"
	const mnistfilelabel = "train-labels.idx1-ubyte"
	const mnistimage = "train-images.idx3-ubyte"

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
	fmt.Println(len(batchedoutput))
	fmt.Println(len(batchesofinputbatches))
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
