package main

import (
	"github.com/dereklstinson/GoCuNets/layers/reshape"
	"fmt"
	"math/rand"
	"time"

	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/gocudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/testing/mnist/mnistgpu"
	"github.com/dereklstinson/GoCuNets/testing/mnistGAN/gand"
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

	for i := 0; i < epocs; i++ {

		for j := 0; j < batchnum; j++ {
			desctrain(handle, descrimintor, gputrainingdata[0], fakelabels, batchsize)
			gentrain(handle, generator, descrimintor, genrandomgpu[j], gputrainingdata[j], reallabels, batchsize)
		}

	}

}
func genoutput(handle *gocunets.Handles, generator *gocunets.Network, randomx,  batch int){
	gy, err := randomx.ZeroClone()
	cherror(err)
	cherror(generator.ForwardProp(handle, nil, randomx, gy)
	gy.I
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
func lossgenerator(dFake,aFake,dReal,aReal []float32,batchsize,classificationsize int){
	fakepercent,fakeloss:=softmaxbatch(aFake,dFake,batchsize,classificationsize)
	realpercent,fakepercent:=softmaxbatch(aReal,dReal,batchsize,classificationsize)
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

