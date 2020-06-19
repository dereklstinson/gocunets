package main

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/dereklstinson/gocudnn/cudart"
	"github.com/dereklstinson/gocunets"
	"github.com/dereklstinson/gocunets/testing/mnist/dfuncs"
	//	"github.com/dereklstinson/gocunets/gocudnn/tensor"
	//	gocudnn "github.com/dereklstinson/gocudnn"
)

func filldatabuffer(target, data []float32, labeled []dfuncs.LabeledData, batchsize int32) {
	randomlabeled := make([]dfuncs.LabeledData, batchsize)
	for i := range randomlabeled {
		randomlabeled[i] = labeled[rand.Int()%len(labeled)]
	}
	for i, rl := range randomlabeled {
		for j := range rl.Data {
			data[i*28*28+j] = rl.Data[j]
		}
		for j := range rl.Label {
			target[i*10+j] = rl.Label[j]
		}

	}
}

//This will make a file with loss timings and losses for each epoch and run.
func main() {
	var learningrate = float32(.005)
	decay1, decay2 := float32(0.0001), float32(0.001)
	args := os.Args[1:]
	var err error
	var devnum = int(1)
	var bsize = 1000
	var savedir string
	epochs := 200
	convergedtestloss := float32(.01)
	if len(args) > 0 {
		devnum, err = strconv.Atoi(args[0])
		cherror(err)
	}
	if len(args) > 1 {
		bsize, err = strconv.Atoi(args[1])
		cherror(err)
	}
	if len(args) > 2 {
		savedir = args[2]
		if !strings.HasSuffix(savedir, "/") {
			savedir = savedir + "/"
		}
	}
	runtime.LockOSThread()
	var nruns = 1
	for nrunindex := 0; nrunindex < nruns; nrunindex++ {
		batchsize := int32(bsize)

		rand.Seed(time.Now().UnixNano())
		//	savelocationforimages := "/home/derek/Desktop/GANMNIST/"
		//	imagenames := "MNIST"
		//trainingkernellocation := "/home/derek/go/src/github.com/dereklstinson/gocudnn/kernels/"

		devices, err := gocunets.GetDeviceList()
		cherror(err)
		if devnum >= len(devices) {
			fmt.Printf("\nDevnum too hight must be 0 through %v\n", len(devices)-1)
			return
		}
		device := devices[devnum]
		err = device.Set()
		cherror(err)
		w := gocunets.CreateWorker(device)
		handle := gocunets.CreateHandle(w, device, 32)
		stream, err := cudart.CreateNonBlockingStream()
		cherror(handle.SetStream(stream))
		builder := gocunets.CreateBuilder(handle)
		builder.Cmode.CrossCorrelation()
		builder.AMode.Leaky()
		builder.Dtype.Float()
		builder.Frmt.NCHW()
		builder.Mtype.Default()
		builder.Nan.NotPropigate()
		/*
		   Lets go ahead and start loading the training data

		*/
		//asdfas
		filetrainbatchloss, err := os.Create(savedir + strconv.Itoa(int(batchsize)) + "TrainBatchLoss")
		cherror(err)
		defer filetrainbatchloss.Close()
		filetestbatchloss, err := os.Create(savedir + strconv.Itoa(int(batchsize)) + "TestBatchLoss")
		cherror(err)
		defer filetestbatchloss.Close()
		fileEpocLossTime, err := os.Create(savedir + strconv.Itoa(int(batchsize)) + "TestTrainEpocLossTime")
		cherror(err)
		defer fileEpocLossTime.Close()
		fmt.Fprintf(fileEpocLossTime, "%10s,\t%10s,\t%10s,\t%10s\t\n", "\"Epoch\"", "\"TrainLoss\"", "\"TestLoss\"", "\"Time(s)\"")
		fmt.Fprintf(filetrainbatchloss, "%10s\n", "\"Train\"")
		fmt.Fprintf(filetestbatchloss, "%10s\n", "\"Test\"")
		const filedirectory = "../mnist/files/"
		const mnistfilelabeltrain = "train-labels.idx1-ubyte"
		const mnistimagetrain = "train-images.idx3-ubyte"
		const mnistfilelabeltest = "t10k-labels.idx1-ubyte"
		const mnistimagetest = "t10k-images.idx3-ubyte"

		mnistdatatest, err := dfuncs.LoadMNIST(filedirectory, mnistfilelabeltest, mnistimagetest)
		mnistdatatrain, err := dfuncs.LoadMNIST(filedirectory, mnistfilelabeltrain, mnistimagetrain)

		var dtaverageimage float32
		//	var wg sync.WaitGroup
		//	wg.Add(1)
		//	go func() {
		for _, dt := range mnistdatatest {
			var dtadder float32
			for i := range dt.Data {
				dtadder = dtadder + dt.Data[i]
			}
			dtaverageimage = dtaverageimage + dtadder/float32(len(dt.Data))
		}
		//		wg.Done()
		//	}()

		var dtrainaverageimage float32
		for _, dt := range mnistdatatrain {
			var dtadder float32
			for i := range dt.Data {
				dtadder = dtadder + dt.Data[i]
			}
			dtrainaverageimage = dtaverageimage + dtadder/float32(len(dt.Data))
		}
		//	wg.Wait()

		totalaverage := dtaverageimage/float32(len(mnistdatatest)) + dtrainaverageimage/float32(len(mnistdatatrain))

		mnistdatatest = dfuncs.NormalizeData(mnistdatatest, totalaverage)

		mnistdatatrain = dfuncs.NormalizeData(mnistdatatrain, totalaverage)

		fmt.Println("Average Pixel is", totalaverage)
		fmt.Println("TrainingSize", len(mnistdatatrain))
		fmt.Println("Testing Size", len(mnistdatatest))
		mnet := gocunets.CreateSimpleModuleNetwork(0, builder)

		ntrainbatches := int32(len(mnistdatatrain)) / batchsize
		ntestbatches := int32(len(mnistdatatest)) / batchsize
		inputdims := []int32{batchsize, 1, 28, 28}
		fmt.Println("Train,Test Number of Batches", ntrainbatches, ntestbatches)

		mods := make([]gocunets.Module, 3)
		mods[0], err = gocunets.CreateVanillaModule(0, builder, batchsize, []int32{20, 1, 4, 4}, []int32{6, 6}, []int32{2, 2}, []int32{3, 3}, 1, 0, 1, 0)
		if err != nil {
			panic(err)
		}
		mods[1], err = gocunets.CreateVanillaModule(1, builder, batchsize, []int32{20, 20, 4, 4}, []int32{4, 4}, []int32{2, 2}, []int32{3, 3}, 1, 0, 1, 0)
		if err != nil {
			panic(err)
		}
		mods[2], err = gocunets.CreateVanillaModule(2, builder, batchsize, []int32{20, 20, 4, 4}, []int32{4, 4}, []int32{2, 2}, []int32{3, 3}, 1, 0, 1, 0)
		if err != nil {
			panic(err)
		}

		channeladder := int32(20)

		mnet.SetModules(mods)
		InputTensor, err := builder.CreateTensor(inputdims)
		if err != nil {
			panic(err)
		}
		mnet.SetTensorX(InputTensor)
		outputdims, err := mnet.FindOutputDims()
		if err != nil {
			panic(err)
		}
		//THis has to be NCHW
		fmt.Println("OutputDims", outputdims)

		outputfdims := []int32{10, channeladder, outputdims[2], outputdims[3]}
		mnet.Output, err = gocunets.CreateOutputModule(5, builder, batchsize, outputfdims, []int32{0, 0}, []int32{1, 1}, []int32{1, 1}, 1, 0, 1, 0)
		if err != nil {
			panic(err)
		}
		err = mnet.SetSoftMaxClassifier()
		if err != nil {
			panic(err)
		}
		outputdims, err = mnet.FindOutputDims()
		if err != nil {
			panic(err)
		}
		fmt.Println("NewOutputDims", outputdims)
		ohy, err := builder.CreateTensor(outputdims)
		if err != nil {
			panic(err)
		}
		mnet.Classifier.SetTensorY(ohy)
		ohdy, err := builder.CreateTensor(outputdims)
		if err != nil {
			panic(err)
		}
		mnet.Classifier.SetTensorDY(ohdy)

		err = mnet.InitHiddenLayers()
		if err != nil {
			panic(err)
		}
		err = mnet.InitWorkspace()
		if err != nil {
			panic(err)
		}
		allmnetweights := mnet.GetWeights()
		allmnetdeltaweights := mnet.GetDeltaWeights()
		trainhandler := gocunets.CreateTrainerHandler(builder)
		adamtrainers, err := trainhandler.SetupAdamTrainers(allmnetweights, allmnetdeltaweights, learningrate, decay1, decay2, batchsize)
		if err != nil {
			panic(err)
		}
		fmt.Println("Number of trainers is", len(adamtrainers))
		var mux sync.Mutex

		trainloss := make([]float32, ntrainbatches)
		testloss := make([]float32, ntestbatches)

		//	for i := range testdatabuffers {
		//		filldatabuffer(testtargetbuffers[i], testdatabuffers[i], mnistdatatest, batchsize)
		//	}
		testtensors := make([]*gocunets.Tensor, ntestbatches)
		testtensorstarget := make([]*gocunets.Tensor, ntestbatches)
		for i := range testtensors {
			testtensors[i], err = builder.CreateTensor([]int32{batchsize, 1, 28, 28})
			cherror(err)
			testtensorstarget[i], err = builder.CreateTensor([]int32{batchsize, 10, 1, 1})
			cherror(err)

		}
		trainingtensors := make([]*gocunets.Tensor, ntrainbatches)
		traintargettesnors := make([]*gocunets.Tensor, ntrainbatches)
		for i := range trainingtensors {
			trainingtensors[i], err = builder.CreateTensor([]int32{batchsize, 1, 28, 28})
			cherror(err)
			traintargettesnors[i], err = builder.CreateTensor([]int32{batchsize, 10, 1, 1})
			cherror(err)
		}

		for i := int32(0); i < ntrainbatches; i++ {
			var trainingfloats []float32
			var trainingtargets []float32
			for j := int32(0); j < batchsize; j++ {
				position := rand.Int63() % (int64)(len(mnistdatatrain))
				trainingfloats = append(trainingfloats, mnistdatatrain[position].Data...)
				trainingtargets = append(trainingtargets, mnistdatatrain[position].Label...)

			}
			cherror(trainingtensors[i].LoadValuesFromSLice(handle.Handler, trainingfloats, (int32)(len(trainingfloats))))
			cherror(traintargettesnors[i].LoadValuesFromSLice(handle.Handler, trainingtargets, (int32)(len(trainingtargets))))
		}
		for i := int32(0); i < ntestbatches; i++ {
			var testfloats []float32
			var testtargets []float32
			for j := int32(0); j < batchsize; j++ {
				position := rand.Int63() % (int64)(len(mnistdatatest))
				testfloats = append(testfloats, mnistdatatest[position].Data...)
				testtargets = append(testtargets, mnistdatatest[position].Label...)
			}
			cherror(testtensors[i].LoadValuesFromSLice(handle.Handler, testfloats, (int32)(len(testfloats))))
			cherror(testtensorstarget[i].LoadValuesFromSLice(handle.Handler, testtargets, (int32)(len(testtargets))))
		}

		//	filldatabuffer(traintargetbuffers[i], traindatabuffers[i], mnistdatatrain, batchsize)

		var donessignal bool
		fmt.Printf("%10s,\t%10s,\t%10s,\t%10s\t\n", "\"Epoch\"", "\"TrainLoss\"", "\"TestLoss\"", "\"Time(s)\"")
		for k := 0; k < epochs; k++ {
			timer := time.Now()
			rand.Shuffle(len(trainingtensors), func(i, j int) {
				trainingtensors[i], trainingtensors[j] = trainingtensors[j], trainingtensors[i]
				traintargettesnors[i], traintargettesnors[j] = traintargettesnors[j], traintargettesnors[i]
			})
			batchratio := ntrainbatches / ntestbatches
			if donessignal {

				break
			}
			for j := int32(0); j < ntestbatches; j++ {
				if donessignal {

					break
				}
				for l := j * batchratio; l < j*batchratio+batchratio; l++ {
					cherror(InputTensor.LoadMem(handle.Handler, trainingtensors[l], trainingtensors[l].SIB()))
					cherror(mnet.GetTensorDY().LoadMem(handle.Handler, traintargettesnors[l], traintargettesnors[l].SIB()))
					cherror(stream.Sync())
					cherror(mnet.Forward())
					cherror(stream.Sync())
					cherror(stream.Sync())
					cherror(mnet.Backward())
					cherror(stream.Sync())
					loss := mnet.GetLoss()
					if math.IsNaN(float64(loss)) {
						panic(trainloss)
					}
					trainloss[l] = loss

					cherror(stream.Sync())
				}
				cherror(trainhandler.UpdateWeights())
				cherror(InputTensor.LoadMem(handle.Handler, testtensors[j], testtensors[j].SIB()))
				cherror(mnet.GetTensorDY().LoadMem(handle.Handler, testtensorstarget[j], testtensorstarget[j].SIB()))
				cherror(mnet.TestForward())
				cherror(stream.Sync())
				testloss[j] = mnet.GetLoss()
			}

			if donessignal {

				break
			}

			cherror(stream.Sync())
			timetoepoch := float32(time.Now().Sub(timer).Seconds())
			if k < epochs-1 {

				go func(k int, timetoepoch float32, fileEpocLossTime, filetestbatchloss, filetrainbatchloss io.Writer) {

					testlosscopy := make([]float32, ntestbatches)
					trainlosscopy := make([]float32, ntrainbatches)
					mux.Lock()
					copy(trainlosscopy, trainloss)
					copy(testlosscopy, testloss)
					mux.Unlock()

					var trainlossadder float32
					var testlossadder float32
					for i := range trainlosscopy {
						trainlossadder += trainlosscopy[i]
					}
					trainlossadder /= float32(len(trainlosscopy))
					for i := range testlosscopy {
						testlossadder += testlosscopy[i]
					}
					testlossadder /= float32(len(testlosscopy))
					fmt.Printf("%10d,\t%10.5f,\t%10.5f,\t%10.5f\n", k, trainlossadder, testlossadder, timetoepoch)
					//	percent, loss := epocoutputchecker(netoutput, desiredoutput, testbatchnum, batchsize, 10)
					fmt.Fprintf(fileEpocLossTime, "%10d,\t%10.5f,\t%10.5f,\t%10.5f\n", k, trainlossadder, testlossadder, timetoepoch) //sizes of strings 5,9,8,7
					go func(trainlosscopy []float32) {
						for i := range trainlosscopy {
							fmt.Fprintf(filetrainbatchloss, "%10.5f\n", trainlosscopy[i])
						}
					}(trainlosscopy)
					go func(testlosscopy []float32) {
						for i := range testlosscopy {
							fmt.Fprintf(filetestbatchloss, "%10.5f\n", testlosscopy[i])
						}
					}(testlosscopy)
					if testlossadder < convergedtestloss {
						donessignal = true
					}

				}(k, timetoepoch, fileEpocLossTime, filetestbatchloss, filetrainbatchloss)

			} else {
				testlosscopy := make([]float32, ntestbatches)
				trainlosscopy := make([]float32, ntrainbatches)
				mux.Lock()
				copy(trainlosscopy, trainloss)
				copy(testlosscopy, testloss)
				mux.Unlock()
				var trainlossadder float32
				var testlossadder float32
				for i := range trainlosscopy {
					trainlossadder = trainlossadder + trainlosscopy[i]
				}
				trainlossadder /= float32(len(trainlosscopy))
				for i := range testlosscopy {
					testlossadder = testlossadder + testlosscopy[i]
				}
				testlossadder /= float32(len(testlosscopy))
				fmt.Printf("%10d,\t%10.5f,\t%10.5f,\t%10.5f\n", k, trainlossadder, testlossadder, timetoepoch)
				//	percent, loss := epocoutputchecker(netoutput, desiredoutput, testbatchnum, batchsize, 10)
				fmt.Fprintf(fileEpocLossTime, "%10d,\t%10.5f,\t%10.5f,\t%10.5f\n", k, trainlossadder, testlossadder, timetoepoch) //sizes of strings 5,9,8,7
				var wg sync.WaitGroup
				wg.Add(1)
				go func(trainlosscopy []float32) {
					for i := range trainlosscopy {
						fmt.Fprintf(filetrainbatchloss, "%10.5f\n", trainlosscopy[i])
					}
					wg.Done()
				}(trainlosscopy)
				wg.Add(1)
				go func(testlosscopy []float32) {
					for i := range testlosscopy {
						fmt.Fprintf(filetestbatchloss, "%10.5f\n", testlosscopy[i])
					}
					wg.Done()
				}(testlosscopy)
				wg.Wait()

			}

		}
	}

	runtime.UnlockOSThread()

}
func printoutput(numofans, batchsize int, input []float32) {
	for i := 0; i < batchsize; i++ {
		for j := 0; j < numofans; j++ {
			fmt.Printf("%-0.2f ", input[i*numofans+j])
		}
		fmt.Printf("\n ")
	}
}
func epocoutputchecker(actual, desired [][]float32, batchtotal, batchsize, classificationsize int) (float64, float64) {
	var batchloss float64
	var percent float64
	for i := 0; i < batchtotal; i++ {
		perc, batch := batchoutputchecker(actual[i], desired[i], batchsize, classificationsize)
		batchloss += batch
		percent += perc
	}
	return percent / float64(batchtotal), batchloss / float64(batchtotal)

}
func batchoutputchecker(actual, desired []float32, batchsize, classificationsize int) (float64, float64) {
	var batchloss float64
	var percent float64
	var position int
	//	delta := float64(-math.Log(float64(output.SoftOutputs[i]))) * desiredoutput[i]
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
				value := (-math.Log(float64(actual[ijpos])))
				if math.IsInf(float64(value), 0) == true {
					fmt.Println("Output Value: ", value)
				}
				batchloss += value
			}

		}
		percent += float64(desired[position])

	}
	if math.IsNaN(float64(batchloss)) == true {
		panic("reach NAN")
	}

	return percent / float64(batchsize), batchloss / float64(batchsize)
}

func dims(args ...int) []int32 {

	length := len(args)
	x := make([]int32, length)
	for i := 0; i < length; i++ {
		x[i] = int32(args[i])
	}
	return x
}
func cherror(input error) {
	if input != nil {
		fmt.Println("***************************")
		panic(input)

	}
}
func getsize(dims []int32) int32 {
	mult := int32(1)
	for i := range dims {
		mult *= dims[i]
	}
	return mult

}
