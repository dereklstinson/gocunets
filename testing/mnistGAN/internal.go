package mnistGAN

import (
	"fmt"
	"math/rand"

	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

func makefakereallabels(smooth, real bool, input *layers.IO, dims []int32, frmt gocudnn.TensorFormat, dtype gocudnn.DataType) (*layers.IO, error) {
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

	labels := generatelabelsmoothingtensor(real, int(dims[0]))

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

type tensor struct {
	data []float32
	dims []int32
}

func cherror(input error) {
	if input != nil {
		fmt.Println("***************************")
		panic(input)

	}
}

//Gaussian returns the gaussien at zero
func gaussianstd3333() float32 {
	return float32(utils.Gaussian(0, .3333))
}

func dims(args ...int) []int32 {

	length := len(args)
	x := make([]int32, length)
	for i := 0; i < length; i++ {
		x[i] = int32(args[i])
	}
	return x
}

//MakeRandomGaussianTensor makes a random gaussian tensor with the std of .33333
func makerandomgaussiantensor(amount int, dims []int32) []tensor {
	size := 1
	for i := 0; i < len(dims); i++ {
		size *= int(dims[i])

	}
	tens := make([]tensor, amount)
	for i := range tens {
		tens[i].data = make([]float32, size)
		tens[i].dims = dims
		for j := range tens[i].data {
			tens[i].data[j] = gaussianstd3333()
		}
	}

	return tens

}
