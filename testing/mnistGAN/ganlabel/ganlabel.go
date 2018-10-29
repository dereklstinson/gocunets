package ganlabel

import (
	"math/rand"

	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//MakeFakeRealLabels is a simple classification 01 or 01
//This doesn't seem to work
func MakeFakeRealLabels(smooth, real bool, input *layers.IO, dims []int32, frmt gocudnn.TensorFormat, dtype gocudnn.DataType) (*layers.IO, error) {
	if input == nil {
		if smooth == true {
			labels := generatelabelsmoothingtensor(real, int(dims[0]))
			layersio, err := layers.BuildIO(frmt, dtype, dims, true)
			if err != nil {
				return nil, err
			}
			ptr, err := gocudnn.MakeGoPointer(labels)
			if err != nil {
				return nil, err
			}
			err = layersio.LoadDeltaTValues(ptr)
			return layersio, err
		}
		labels := generatelabeltensors(real, int(dims[0]))
		layersio, err := layers.BuildIO(frmt, dtype, dims, true)
		if err != nil {
			return nil, err
		}
		ptr, err := gocudnn.MakeGoPointer(labels)
		if err != nil {
			return nil, err
		}
		err = layersio.LoadDeltaTValues(ptr)
		return layersio, err
	}
	if smooth == true {
		labels := generatelabelsmoothingtensor(real, int(dims[0]))

		ptr, err := gocudnn.MakeGoPointer(labels)
		if err != nil {
			return nil, err
		}
		err = input.LoadDeltaTValues(ptr)
		return input, err
	}
	labels := generatelabeltensors(real, int(dims[0]))

	ptr, err := gocudnn.MakeGoPointer(labels)
	if err != nil {
		return nil, err
	}
	err = input.LoadDeltaTValues(ptr)
	return input, err
}

func generatelabelsmoothingtensor(real bool, batches int) []float32 {
	const randomstartposition = float32(.7)
	const randommultiplier = float32(.5)
	holder := make([]float32, 0)

	if real == true {

		for i := 0; i < batches; i++ {
			value := randomstartposition + rand.Float32()*randommultiplier

			real := []float32{value, 0}
			holder = append(holder, real...)
		}
	} else {
		value := randomstartposition + rand.Float32()*randommultiplier
		fake := []float32{0, value}
		for i := 0; i < batches; i++ {
			holder = append(holder, fake...)
		}
	}
	gpuusable := make([]float32, len(holder))
	for i := range gpuusable {
		gpuusable[i] = holder[i]
	}
	return gpuusable
}
func generatelabeltensors(real bool, batches int) []float32 {
	holder := make([]float32, 0)

	if real == true {
		real := []float32{1, 0}

		for i := 0; i < batches; i++ {
			holder = append(holder, real...)
		}
	} else {
		fake := []float32{0, 1}
		for i := 0; i < batches; i++ {
			holder = append(holder, fake...)
		}
	}
	gpuusable := make([]float32, len(holder))
	for i := range gpuusable {
		gpuusable[i] = holder[i]
	}
	return gpuusable
}
