package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/dereklstinson/GoCuNets/layers"

	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/gocudnn/convolution"

	"github.com/dereklstinson/GoCuNets/testing/mnist/mnistgpu"
	"github.com/dereklstinson/GoCuNets/testing/mnistGAN/gand"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

func main() {
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

	NanProp := gocudnn.PropagationNANFlag{}.NotPropagateNan()
	memmanaged := true
	AMode := gocudnn.ActivationModeFlag{}.Relu()

	batchsize := 20 // how many forward and backward runs before updating weights.

	gputrainingdata, gpuanswersdata, gputestingdata, gputestansdata := mnistgpu.MNISTGpuLabels(batchsize, frmt, dtype, memmanaged)
	batchnum := len(gputrainingdata)
	testbatchnum := len(gputestingdata)
	genrandom := makerandomgaussiantensor(batchnum, []int32{int32(batchsize), 1, 28, 28})
	genrandomgpu := make([]*layers.IO, 0)
	for i := range genrandom {
		//genrandom[i]
		uploaded, err := layers.BuildNetworkInputIO(frmt, dtype, genrandom[i].dims, memmanaged)
		cherror(err)
		randcpuptr, err := gocudnn.MakeGoPointer(uploaded)
		cherror(err)
		err = uploaded.LoadTValues(randcpuptr)
		cherror(err)
		genrandomgpu = append(genrandomgpu, uploaded)
	}
	generator := gand.Generator(handle, frmt, dtype, CMode, AMode, memmanaged, batchsize)
	descrimintor := gand.Descriminator(handle, frmt, dtype, CMode, AMode, memmanaged, batchsize)
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
	return (.33333 * gauassian())
}

func gauassian() float32 {
	//Polar method
	var x, y, z float64
	for z >= 1 || z == 0 {
		x = (2 * rand.Float64()) - float64(1)
		y = (2 * rand.Float64()) - float64(1)
		z = x*x + y*y
	}

	return float32(x * math.Sqrt(-2*math.Log(z)/z))
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
func minGmaxD(DGz []float32, Dx []float32) []float32 {
	return []float32{}
}
