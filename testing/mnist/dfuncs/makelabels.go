package dfuncs

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

const numLabels = 10
const pixelRange = 255

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
	Width      = 28
	Height     = 28
)

type LabeledData struct {
	Data  []float32
	Label []float32
}

func LoadMNIST(filedirectory string, filenameLabel string, filenameData string) ([]LabeledData, error) {

	labelfile, err := os.Open(filedirectory + filenameLabel)
	if err != nil {
		//	panic(err)
		return nil, err
		//	panic(err)
	}
	alllabels, err := readLabelFile(labelfile)
	if err != nil {
		//	panic(err)
		return nil, err
	}
	datafile, err := os.Open(filedirectory + filenameData)
	if err != nil {
		//	panic(err)
		return nil, err
	}
	alldata, err := readImageFile(datafile)
	if err != nil {
		//	panic(err)
		return nil, err
	}
	if len(alldata) != len(alllabels) {
		return nil, errors.New("datafile and label file lengths don't match")
	}
	labeled := make([]LabeledData, len(alldata))
	for i := 0; i < len(alldata); i++ {
		labeled[i].Data = alldata[i]
		labeled[i].Label = alllabels[i]
	}
	return labeled, nil
}

func readLabelFile(r io.Reader) ([][]float32, error) {
	var err error

	var (
		magic int32
		n     int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		//panic(err)
		return nil, err
	}
	if magic != labelMagic {
		fmt.Println(magic, labelMagic)
		return nil, os.ErrInvalid
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	labels := make([][]float32, n)
	for i := 0; i < int(n); i++ {
		var l uint8
		if err := binary.Read(r, binary.BigEndian, &l); err != nil {
			return nil, err
		}
		labels[i] = append(labels[i], makeonehotstate(l)...)
	}
	return labels, nil
}
func NormalizeData(data []LabeledData, average float32) []LabeledData {
	size := len(data)
	for i := 0; i < size; i++ {
		for j := 0; j < len(data[i].Data); j++ {
			data[i].Data[j] = data[i].Data[j] / average
		}

	}
	return data
}
func FindAverage(input []LabeledData) float32 {
	inputsize := len(input)
	datasize := len(input[0].Data)
	var adder float32
	for i := 0; i < inputsize; i++ {

		for j := 0; j < datasize; j++ {
			adder += input[i].Data[j]
		}
	}
	return adder / float32(inputsize*datasize)
}
func makeonehotstate(input uint8) []float32 {
	x := make([]float32, 10)
	x[input] = float32(1.0)
	return x

}
func readImageFile(r io.Reader) ([][]float32, error) {

	var err error

	var (
		magic int32
		n     int32
		nrow  int32
		ncol  int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != imageMagic {
		return nil, err /*os.ErrInvalid*/
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &nrow); err != nil {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &ncol); err != nil {
		return nil, err
	}
	imgflts := make([][]float32, n)
	imgs := make([][]byte, n)
	size := int(nrow * ncol)
	for i := 0; i < int(n); i++ {
		imgflts[i] = make([]float32, size)
		imgs[i] = make([]byte, size)

		actual, err := io.ReadFull(r, imgs[i])
		if err != nil {
			return nil, err
		}
		if size != actual {
			return nil, os.ErrInvalid
		}
		for j := 0; j < size; j++ {
			imgflts[i][j] = float32(imgs[i][j])
		}
	}

	return imgflts, nil
}
