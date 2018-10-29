package dfuncs

import (
	"errors"
	"os"
)

//LoadMNIST11gan will add an extra zero at the end so that it has a fake class
func LoadMNIST11gan(filedirectory string, filenameLabel string, filenameData string) ([]LabeledData, error) {

	labelfile, err := os.Open(filedirectory + filenameLabel)
	if err != nil {
		//	panic(err)
		return nil, err
		//	panic(err)
	}
	alllabels, numbers, err := readLabelFile(labelfile)
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
		alllabels[i] = append(alllabels[i], 0)
		labeled[i].Data = alldata[i]
		labeled[i].Label = alllabels[i]
		labeled[i].Number = numbers[i]
	}
	return labeled, nil
}
