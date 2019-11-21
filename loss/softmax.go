package loss

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/reduce"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/utils"
	"github.com/dereklstinson/GoCudnn/gocu"

	"math"
)

//SoftMax Holds the methods to do softmax loss
type SoftMax struct {
	reducetensor *tensor.Volume
	hostmem      []float32
	hostptr      *gocu.Wrapper
	wspace       *nvidia.Malloced
	op           *reduce.Ops
	wspacesib    uint
}

//MakeSoftMaxLossCalculator returns a loss calculator for softmax
func MakeSoftMaxLossCalculator() SoftMax {
	return SoftMax{}
}

//BatchLoss takes the actual and desired arrays in the form of i=batchindex, j=classindex actual[i*classificationsize+j]
func (s SoftMax) BatchLoss(actual, desired []float32, batchsize, classificationsize int) (percent, loss float32) {
	percent, loss = s.batchlossandpercent(actual, desired, batchsize, classificationsize)
	return percent, loss
}

//EpocLossFromBatchLosses takes an arrays of percent and loss accumulated over the batches and returns total loss over those batches
func (s SoftMax) EpocLossFromBatchLosses(percentb, lossb []float32) (percent, loss float32) {
	padder := float32(0)
	ladder := float32(0)
	length := len(percentb)
	for i := 0; i < length; i++ {
		padder += percentb[i]
		ladder += lossb[i]
	}
	percent = padder / float32(length)
	loss = ladder / float32(length)
	return percent, loss
}

//EpocLoss returns the loss epoc if BatchLoss was not calculated.
func (s SoftMax) EpocLoss(actual, desired [][]float32, batchsize, classificationsize int) (percent, loss float32) {

	percent, loss = s.lossonepoc(actual, desired, batchsize, classificationsize)
	return percent, loss

}
func (s SoftMax) lossonepoc(actual, desired [][]float32, batchsize, classificationsize int) (float32, float32) {
	var batchloss float32
	var percent float32
	batchtotal := len(actual)
	for i := 0; i < batchtotal; i++ {
		perc, batch := s.batchlossandpercent(actual[i], desired[i], batchsize, classificationsize)
		batchloss += batch
		percent += perc
	}
	return percent / float32(batchtotal), batchloss / float32(batchtotal)

}

func (s SoftMax) batchlossandpercent(actual, desired []float32, numofbatches, classificationsize int) (float32, float32) {
	var batchloss float32
	var percent float32
	var position int
	//	delta := float64(-math.Log(float64(output.SoftOutputs[i]))) * desiredoutput[i]
	for i := 0; i < numofbatches; i++ {

		maxvalue := float32(-99999)
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
	return percent / float32(numofbatches), batchloss / float32(numofbatches)
}

func (s *SoftMax) SoftMaxLoss(h *cudnn.Handler, target, actual *layers.IO) (float32, error) {
	var err error
	if s.op == nil {
		frmt, dtype, dims, err := target.Properties()
		if err != nil {
			return 0, err
		}
		frmtflg := frmt
		if frmt == frmtflg.NCHW() {
			for i := 2; i < len(dims); i++ {
				dims[i] = 1
			}
		} else {
			for i := 1; i < len(dims)-1; i++ {
				dims[i] = 1
			}
		}
		s.hostmem = make([]float32, utils.FindVolumeInt32(dims, nil))
		s.hostptr, err = gocu.MakeGoMem(s.hostmem)
		if err != nil {
			return 0, err
		}
		err = actual.T().OpMult(h, actual.T(), target.T(), 1, 1, 0)
		s.reducetensor, err = tensor.Build(h, frmt, dtype, dims)
		s.op, err = reduce.Stage(reduce.Flags.ReduceMode.MulNoZeros(), reduce.Flags.DType.Float(), reduce.Flags.NanProp.NotPropigate(), reduce.Flags.IndFlag.NoIndices(), reduce.Flags.IndType.Type32Bit())
		if err != nil {
			return 0, err
		}
		s.wspacesib, err = s.op.GetWorkSpaceSize(h, actual.T(), s.reducetensor)
		if err != nil {
			return 0, err
		}
		s.wspace, err = nvidia.MallocGlobal(h, s.wspacesib)
		if err != nil {
			return 0, err
		}
		err = s.op.Reduce(h, nil, s.wspace, 1, actual.T(), 0, s.reducetensor)
		if err != nil {
			return 0, err
		}
		h.Sync()
		s.reducetensor.FillSlice(s.hostmem, (int32)(len(s.hostmem)))
		h.Sync()
		var storage float64
		for i := range s.hostmem {
			storage += math.Log(float64(s.hostmem[i]))
		}
		return float32(-storage), nil
	}

	err = actual.T().OpMult(h, actual.T(), target.T(), 1, 1, 0)
	err = s.op.Reduce(h, nil, s.wspace, 1, actual.T(), 0, s.reducetensor)
	if err != nil {
		return 0, err
	}

	h.Sync()
	s.reducetensor.FillSlice(s.hostmem, (int32)(len(s.hostmem)))
	h.Sync()
	var storage float64
	for i := range s.hostmem {
		storage += math.Log(float64(s.hostmem[i]))
	}
	return float32(-storage), nil
}
