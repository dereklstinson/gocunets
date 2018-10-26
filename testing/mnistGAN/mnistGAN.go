package main

import (
	"fmt"
	"image"
	"math"
	"math/rand"
	"strconv"
	"time"

	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/gocudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/testing/mnist/mnistgpu"
	"github.com/dereklstinson/GoCuNets/testing/mnistGAN/gand"
	"github.com/dereklstinson/GoCuNets/utils"
	"github.com/dereklstinson/GoCuNets/utils/filing"
	"github.com/dereklstinson/GoCuNets/utils/imaging"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

func main() {
	savelocationforimages := "/home/derek/GANMNIST/"
	imagenames := "MNIST"
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

	gputrainingdata, _ := mnistgpu.WithCPULabels(batchsize, frmt, dtype, memmanaged)
	batchnum := len(gputrainingdata)
	//	testbatchnum := len(gputestingdata)
	genrandomtensors := makerandomgaussiantensor(batchnum, []int32{int32(batchsize), 1, 28, 28})

	genrandomgpu := make([]*layers.IO, 0)
	for i := range genrandomtensors {
		//genrandom[i]
		uploaded, err := layers.BuildNetworkInputIO(frmt, dtype, genrandomtensors[i].dims, memmanaged)
		cherror(err)
		randcpuptr, err := gocudnn.MakeGoPointer(genrandomtensors[i].data)
		cherror(err)
		err = uploaded.LoadTValues(randcpuptr)
		cherror(err)
		genrandomgpu = append(genrandomgpu, uploaded)
	}
	generator := gand.Generator(handle, frmt, dtype, CMode, memmanaged, batchsize)
	descrimintor := gand.Descriminator(handle, frmt, dtype, CMode, memmanaged, batchsize)
	epocs := 200
	fakelabels, err := makefakereallabels(false, false, nil, []int32{int32(batchsize), 2, 1, 1}, frmt, dtype)
	cherror(err)
	reallabels, err := makefakereallabels(false, true, nil, []int32{int32(batchsize), 2, 1, 1}, frmt, dtype)
	cherror(err)
	imager, err := imaging.MakeImager(handle.XHandle())
	cherror(err)

	//	images := make([]image.Image, 0)
	for i := 0; i < epocs; i++ {
		realoutput := make([][]float32, batchnum)
		fakeoutput := make([][]float32, batchnum)
		realdesired := make([][]float32, batchnum)
		fakedesired := make([][]float32, batchnum)
		for j := 0; j < batchnum; j++ {
			realoutput[j] = make([]float32, 2*batchsize)
			fakeoutput[j] = make([]float32, 2*batchsize)
			realdesired[j] = make([]float32, 2*batchsize)
			fakedesired[j] = make([]float32, 2*batchsize)

			desctrain(handle, descrimintor, gputrainingdata[j], reallabels, batchsize)
			reallabels.T().Memer().FillSlice(realoutput[j])
			reallabels.DeltaT().Memer().FillSlice(realdesired[j])
			gentrain(handle, generator, descrimintor, genrandomgpu[0], gputrainingdata[0], fakelabels, batchsize)
			fakelabels.T().Memer().FillSlice(fakeoutput[j])
			fakelabels.DeltaT().Memer().FillSlice(fakedesired[j])
			stream.Sync()
		}
		epocloss(fakedesired, fakeoutput, realdesired, realoutput, batchsize, 2, i)
		img, err := genoutputimage(handle, generator, gputrainingdata[0], imager)
		cherror(err)
		filing.WriteImage(savelocationforimages, imagenames+strconv.Itoa(i), img)
	}

}

func genoutputimage(handle *gocunets.Handles, generator *gocunets.Network, randomx *layers.IO, imager *imaging.Imager) (image.Image, error) {
	gy, err := randomx.ZeroClone()
	defer gy.Destroy()
	cherror(err)
	cherror(generator.ForwardProp(handle, nil, randomx, gy))
	return imager.TileBatches(handle.XHandle(), gy, 5, 4)
}
func desctrain(handle *gocunets.Handles, descriminator *gocunets.Network, x, y *layers.IO, batch int) {

	cherror(descriminator.ForwardProp(handle, nil, x, y))
	cherror(descriminator.BackPropFilterData(handle, nil, x, y))
	cherror(descriminator.UpdateWeights(handle, batch))
}
func gentrain(handle *gocunets.Handles, generator, descriminator *gocunets.Network, randomx, notrandomx, y *layers.IO, batch int) {
	gy, err := randomx.ZeroClone()
	cherror(err)
	cherror(generator.ForwardProp(handle, nil, randomx, gy))

	cherror(descriminator.ForwardProp(handle, nil, gy, y))
	cherror(descriminator.BackPropData(handle, nil, gy, y))
	cherror(generator.BackPropFilterData(handle, nil, notrandomx, gy))
	cherror(gy.Destroy())
	cherror(generator.UpdateWeights(handle, batch))

}

func freeIO(input []*layers.IO) {
	for i := range input {
		input[i].Destroy()
	}
}
func epocloss(dFake, aFake, dReal, aReal [][]float32, batchsize, classificationsize, epoc int) {
	totals := make([]float32, 6)
	for i := 0; i < len(dFake); i++ {
		losses := lossgenerator(dFake[i], aFake[i], dReal[i], aReal[i], batchsize, classificationsize, epoc)
		addto(totals, losses)
	}
	divideall(totals, float32(len(dFake)))
	fmt.Printf("Combined: P: %-0.3f L: %-0.3f;  Generator: P: %-0.3f L: %-0.3f; Descriminator: P: %-0.3f L: %-0.3f\n", totals[0], totals[1], totals[2], totals[3], totals[4], totals[5])
}
func addto(a, b []float32) {
	for i := 0; i < len(a); i++ {
		a[i] += b[i]
	}
}
func divideall(a []float32, b float32) {
	for i := range a {
		a[i] /= b
	}
}

func lossgenerator(dFake, aFake, dReal, aReal []float32, batchsize, classificationsize, epoc int) []float32 {
	fakepercent, fakeloss := softmaxbatch(aFake, dFake, batchsize, classificationsize)
	realpercent, realloss := softmaxbatch(aReal, dReal, batchsize, classificationsize)
	percent := (fakepercent + realpercent) / 2
	loss := (fakeloss + realloss) / 2
	return []float32{percent, loss, realpercent, realloss, fakepercent, fakeloss}
	//	fmt.Printf("Epoch Percent Correct: %-0.3f		 Epoch Loss: %-0.3f              Epoch Number: %d\n", percent, loss, epoc)
}
func softmaxbatch(actual, desired []float32, batchsize, classificationsize int) (float32, float32) {
	var batchloss float32
	var percent float32
	var position int

	for i := 0; i < batchsize; i++ {

		maxvalue := float32(0.0)
		ipos := i * classificationsize
		for j := 0; j < classificationsize; j++ {
			ijpos := ipos + j
			if maxvalue < actual[ijpos] {

				maxvalue = actual[ijpos]
				position = ijpos

			}
			if desired[ijpos] != 0 {

				batchloss += float32(-math.Log(float64(actual[ijpos])))
			}

		}
		percent += desired[position]

	}
	if math.IsNaN(float64(batchloss)) == true {
		panic("reach NAN")
	}
	return percent / float32(batchsize), batchloss / float32(batchsize)
}

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
