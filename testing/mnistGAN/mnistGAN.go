package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/gocudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/testing/mnist/mnistgpu"
	"github.com/dereklstinson/GoCuNets/testing/mnistGAN/gand"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

func main() {

	rand.Seed(time.Now().UnixNano())
	trainingkernellocation := "/home/derek/go/src/github.com/dereklstinson/GoCudnn/kernels/"
	gocudnn.Cuda{}.LockHostThread()
	var cuda gocudnn.Cuda
	devices, err := cuda.GetDeviceList()
	cherror(err)
	devicenum := len(devices)
	fmt.Println("Number of Devices:", devicenum)
	device := devices[0]
	if len(devices) == 2 {
		device = devices[1]
	}

	err = device.Set()
	cherror(err)
	handle := gocunets.CreateHandle(device, trainingkernellocation)
	stream, err := gocudnn.Cuda{}.CreateBlockingStream()
	cherror(handle.SetStream(stream))

	var dtypeflags gocudnn.DataTypeFlag
	var fmtflags gocudnn.TensorFormatFlag
	frmt := fmtflags.NCHW()
	dtype := dtypeflags.Float()
	CMode := convolution.Flags().Mode.CrossCorrelation() //.CrossCorrelation()
	//	AMode := gocudnn.ActivationModeFlag{}.Relu()

	//	NanProp := gocudnn.PropagationNANFlag{}.NotPropagateNan()
	memmanaged := true
	//	AMode := gocudnn.ActivationModeFlag{}.Relu()

	batchsize := 20 // how many forward and backward runs before updating weights.

	gputrainingdata, _ := mnistgpu.MNISTGpuNoLabel(batchsize, frmt, dtype, memmanaged)
	batchnum := len(gputrainingdata)
	//	testbatchnum := len(gputestingdata)
	genrandomtensors := makerandomgaussiantensor(batchnum, []int32{int32(batchsize), 1, 28, 28})

	genrandomgpu := make([]*layers.IO, 0)
	for i := range genrandomtensors {
		//genrandom[i]
		uploaded, err := layers.BuildNetworkInputIO(frmt, dtype, genrandomtensors[i].dims, memmanaged)
		cherror(err)
		randcpuptr, err := gocudnn.MakeGoPointer(uploaded)
		cherror(err)
		err = uploaded.LoadTValues(randcpuptr)
		cherror(err)
		genrandomgpu = append(genrandomgpu, uploaded)
	}
	generator := gand.Generator(handle, frmt, dtype, CMode, memmanaged, batchsize)
	descrimintor := gand.Descriminator(handle, frmt, dtype, CMode, memmanaged, batchsize)
	epocs := 200
	for i := 0; i < epocs; i++ {

		for j := 0; j < batchnum; j++ {
			desctrain(handle, descrimintor, nil, nil, batchsize)
			gentrain(handle, generator, descrimintor, nil, nil, batchsize)
		}

	}

}
func desctrain(handle *gocunets.Handles, descriminator *gocunets.Network, x, y *layers.IO, batch int) {

	cherror(descriminator.ForwardProp(handle, nil, x, y))
	cherror(descriminator.BackPropFilterData(handle, nil, x, y))
	cherror(descriminator.UpdateWeights(handle, batch))
}
func gentrain(handle *gocunets.Handles, generator, descriminator *gocunets.Network, x, y *layers.IO, batch int) {
	gy, err := x.ZeroClone()
	cherror(err)
	cherror(generator.ForwardProp(handle, nil, x, gy))

	cherror(descriminator.ForwardProp(handle, nil, gy, y))
	cherror(descriminator.BackPropData(handle, nil, gy, y))
	cherror(generator.BackPropFilterData(handle, nil, x, gy))
	cherror(gy.Destroy())
	cherror(generator.UpdateWeights(handle, batch))

}

func freeIO(input []*layers.IO) {
	for i := range input {
		input[i].Destroy()
	}
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

//I need to build a different backprop where the generator loss flows through discriminator network (not doing the backprop for weights) and goes into the
//generator..
func objectiveminmax(DGz []float32, Dx []float32) float32 {
	SigmaDx := 0.0
	SigmaGDz := 0.0

	if len(DGz) == len(Dx) {
		for i := range Dx {
			SigmaDx += math.Log(float64(Dx[i]))
			SigmaGDz += math.Log(1.0 - float64(DGz[i]))
		}
		SigmaDx /= float64(len(Dx))
		SigmaGDz /= float64(len(DGz))
		return float32(SigmaDx + SigmaGDz)
	}
	for i := range Dx {
		SigmaDx += math.Log(float64(Dx[i]))
	}
	SigmaDx /= float64(len(Dx))
	return float32(0)
}
func crossentropy(networkans, actualans float32) float32 {
	if actualans >= .7 {
		return -float32(math.Log(float64(networkans)))
	}
	return -float32(math.Log(1.0 - float64(networkans)))
}
