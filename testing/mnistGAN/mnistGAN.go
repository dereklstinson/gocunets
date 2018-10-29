package main

import (
	"fmt"
	"image"
	"math/rand"
	"time"

	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/gocudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/loss"
	"github.com/dereklstinson/GoCuNets/testing/mnist/mnistgpu"
	"github.com/dereklstinson/GoCuNets/testing/mnistGAN/gand"
	"github.com/dereklstinson/GoCuNets/testing/mnistGAN/ganlabel"
	"github.com/dereklstinson/GoCuNets/utils"
	"github.com/dereklstinson/GoCuNets/utils/imaging"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

func main() {
	//	savelocationforimages := "/home/derek/Desktop/GANMNIST1/"
	//	imagenames := "MNIST"
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

	memmanaged := true

	batchsize := 20 // how many forward and backward runs before updating weights.

	gputrainingdata, gputraininganswers, _, _ := mnistgpu.WithLabels11Gan(batchsize, frmt, dtype, memmanaged)
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
	descrimintor := gand.DescriminatorClass11(handle, frmt, dtype, CMode, memmanaged, batchsize)
	epocs := 200
	fakelabels, err := ganlabel.MakeFakeClass11Label([]int32{int32(batchsize), 11, 1, 1}, frmt, dtype, memmanaged)
	cherror(err)

	//imager, err := imaging.MakeImager(handle.XHandle())
	cherror(err)
	smClass := loss.MakeSoftMaxLossCalculator()
	//	images := make([]image.Image, 0)
	for i := 0; i < epocs; i++ {

		dpercentb := make([]float32, batchnum)
		dlossb := make([]float32, batchnum)
		gpercentb := make([]float32, batchnum)
		glossb := make([]float32, batchnum)
		for j := 0; j < batchnum; j++ {
			realoutput := make([]float32, 11*batchsize)
			fakeoutput := make([]float32, 11*batchsize)
			realdesired := make([]float32, 11*batchsize)
			fakedesired := make([]float32, 11*batchsize)

			desctrain(handle, stream, descrimintor, gputrainingdata[j], gputraininganswers[j], batchsize)
			stream.Sync()
			gputraininganswers[j].T().Memer().FillSlice(realoutput)
			gputraininganswers[j].DeltaT().Memer().FillSlice(realdesired)
			gentrain(handle, stream, generator, descrimintor, genrandomgpu[j], fakelabels, batchsize)
			stream.Sync()
			fakelabels.T().Memer().FillSlice(fakeoutput)
			fakelabels.DeltaT().Memer().FillSlice(fakedesired)
			stream.Sync()
			updateweights(handle, generator, descrimintor, batchsize)
			dpercentb[j], dlossb[j] = smClass.BatchLoss(realoutput, realdesired, batchsize, 11)
			gpercentb[j], glossb[j] = smClass.BatchLoss(fakeoutput, fakedesired, batchsize, 11)
		}

		stream.Sync()
		//	imgs, err := imager.TileBatches(handle.XHandle(), genrandomgpu[0], 5, 4)

		//	imgs, err := gettiledimage(handle, stream, generator, genrandomgpu[0], imager, 5, 4)
		stream.Sync()
		cherror(err)
		//	cherror(filing.WriteImage(savelocationforimages, imagenames+strconv.Itoa(i), imgs))
		//	for b := range imgs {
		//	cherror(filing.WriteImage(savelocationforimages, imagenames+strconv.Itoa(i)+"-"+strconv.Itoa(b), imgs[b]))
		//}
		dpercente, dlosse := smClass.EpocLossFromBatchLosses(dpercentb, dlossb)
		gpercente, glosse := smClass.EpocLossFromBatchLosses(gpercentb, glossb)
		labelLossPrint("Training ", gpercente, glosse, dpercente, dlosse, i)
	}

}
func makerandomgaussiantensor(numoftensors int, dims []int32) []tensor {
	vol := utils.FindVolumeInt32(dims)
	tensors := make([]tensor, numoftensors)
	for j := range tensors {
		ten := make([]float32, 0, vol)
		for i := 0; i < int(dims[0]); i++ {
			ten = append(ten, utils.RandomGaussianKernelsInChannels(int(dims[2]), int(dims[3]), int(dims[1]), false)...)
		}
		tensors[j].data = ten
		tensors[j].dims = dims

	}
	return tensors

}
func getbatchimages(handle *gocunets.Handles, generator *gocunets.Network, randomx *layers.IO, imager *imaging.Imager, h, w uint) ([]image.Image, error) {
	gy, err := randomx.ZeroClone()
	defer gy.Destroy()
	cherror(err)
	cherror(generator.ForwardProp(handle, nil, randomx, gy))
	return imager.ByBatches(handle.XHandle(), gy, h, w)
}
func gettiledimage(handle *gocunets.Handles, stream *gocudnn.Stream, generator *gocunets.Network, randomx *layers.IO, imager *imaging.Imager, h, w int) (image.Image, error) {
	gy, err := randomx.ZeroClone()

	cherror(err)
	cherror(generator.ForwardProp(handle, nil, randomx, gy))
	stream.Sync()
	images, err := imager.TileBatches(handle.XHandle(), gy, h, w)
	stream.Sync()
	gy.Destroy()
	return images, err
}

func desctrain(handle *gocunets.Handles, stream *gocudnn.Stream, descriminator *gocunets.Network, x, y *layers.IO, batch int) {

	cherror(descriminator.ForwardProp(handle, nil, x, y))
	stream.Sync()
	cherror(descriminator.BackPropFilterData(handle, nil, x, y))
	stream.Sync()
}
func gentrain(handle *gocunets.Handles, stream *gocudnn.Stream, generator, descriminator *gocunets.Network, randomx, y *layers.IO, batch int) {
	gy, err := randomx.ZeroClone()
	cherror(err)
	cherror(generator.ForwardProp(handle, nil, randomx, gy))

	stream.Sync()

	cherror(descriminator.ForwardProp(handle, nil, gy, y))
	stream.Sync()
	cherror(descriminator.BackPropFilterData(handle, nil, gy, y))
	stream.Sync()
	cherror(generator.BackPropFilterData(handle, nil, randomx, gy))
	stream.Sync()

	gy.Destroy()
}

func updateweights(handle *gocunets.Handles, generator, descriminator *gocunets.Network, batch int) {
	cherror(generator.UpdateWeights(handle, batch))
	cherror(descriminator.UpdateWeights(handle, batch))
}
func freeIO(input []*layers.IO) {
	for i := range input {
		input[i].Destroy()
	}
}

func labelLossPrint(label string, gpercent, gloss, dpercent, dloss float32, epoc int) {
	fmt.Printf(label+" Epoc: %d     Generator:    P: %-0.3f L: %-0.3f;    Descriminator:    P: %-0.3f L: %-0.3f\n", epoc, gpercent, gloss, dpercent, dloss)
}
func printfloat(label, frmt string, float []float32) {
	fmt.Printf(label)
	for i := range float {
		fmt.Printf(frmt, float[i])
	}
	fmt.Printf("\n")
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
func dims(args ...int) []int32 {

	length := len(args)
	x := make([]int32, length)
	for i := 0; i < length; i++ {
		x[i] = int32(args[i])
	}
	return x
}
func printoutput(numofans, batchsize int, input []float32) {
	for i := 0; i < batchsize; i++ {
		for j := 0; j < numofans; j++ {
			fmt.Printf("%-0.2f ", input[i*numofans+j])
		}
		fmt.Printf("\n ")
	}
}
