package ganlabel

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//MakeFakeClass11Label will make a bunch of labels with the output of 00000000001 which would mean fake!
func MakeFakeClass11Label(dims []int32, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, unified bool) (*layers.IO, error) {
	var fflg gocudnn.TensorFormatFlag
	zero := int32(0)
	fakelabel := []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
	if frmt == fflg.NCHW() {
		if len(fakelabel) != int(dims[1]) {
			return nil, errors.New("Channel needs to be size 11")
		}
		labels := make([]float32, dims[0]*dims[1])

		for i := zero; i < dims[0]; i++ {
			for j := zero; j < dims[1]; j++ {
				labels[i*dims[1]+j] = fakelabel[j]
			}
		}
		return layers.BuildNetworkOutputIOFromSlice(frmt, dtype, dims, unified, labels)
	} else if frmt == fflg.NHWC() {
		if len(fakelabel) != int(dims[3]) {
			return nil, errors.New("Channel needs to be size 11")
		}
		labels := make([]float32, dims[0]*dims[3])

		for i := zero; i < dims[0]; i++ {
			for j := zero; j < dims[3]; j++ {
				labels[i*dims[3]+j] = fakelabel[j]
			}
		}
		return layers.BuildNetworkOutputIOFromSlice(frmt, dtype, dims, unified, labels)
	}
	return nil, errors.New("Not supported format")

}
