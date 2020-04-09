package loss

import (
	"math"

	"github.com/dereklstinson/GoCudnn/xtra"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/softmax"
)

//SoftMax Holds the methods to do softmax loss
type SoftMax struct {
	s                   *softmax.Layer
	sml                 *xtra.SofMaxLogLoss
	h                   *cudnn.Handler
	alphaloss, betaloss float64
	//	x, y, target, dx    *layers.Tensor // I named the tensors this to keep me from getting confused while writing the functions
	//reducetensor     *layers.Tensor
	hostmem float32
	//	hostmemwspace           []float32
	//hostmemfp16 []half.Float16
	//hostptr     *gocu.Wrapper
	//	wspace, indicies        *nvidia.Malloced
	//	op                      *reduce.Ops
	//softmaxhostmemworkspace []float32
	//fwa, fwb, bwa, bwb float64
	//wspacesib uint
}

//CreateSoftMax creates the soft max function
func CreateSoftMax(h *cudnn.Handler) (s *SoftMax, err error) {
	s = new(SoftMax)
	s.h = h
	s.sml, err = xtra.NewSoftMaxNegLogLoss(h.XHandle())
	if err != nil {
		return nil, err
	}
	//var sfopmul softmax.OpMultiplier
	//sfopmul.ForwardAlpha = 1
	//sfopmul.ForwardBeta = 0
	//sfopmul.BackwardAlpha = -1
	//sfopmul.BackwardBeta = 0

	s.alphaloss = 1
	s.betaloss = 0
	s.s = softmax.StageAccuratePerChannel(nil)
	s.s.SetForwardScalars(1, 0)
	s.s.SetBackwardScalars(-1, 0)
	//	s.x = x
	//	s.y = y
	//	s.target = target
	//	s.dx = dx
	//	tdims := s.y.Dims()
	//reducedims := make([]int32, len(tdims))
	//for i := range reducedims {
	//	reducedims[i] = 1
	//}

	//reducedims[0] = tdims[0]
	//dtype := y.DataType()
	//s.hostmem = make([]float32, reducedims[0])
	////s.hostmemwspace = make([]float32, reducedims[0])
	//switch y.DataType() {
	//case dtype.Float():
	//	s.hostptr, err = gocu.MakeGoMem(s.hostmem)
	//	if err != nil {
	//		return nil, err
	//	}
	//case dtype.Half():
	//	s.hostmemfp16 = make([]half.Float16, reducedims[0])
	//	s.hostptr, err = gocu.MakeGoMem(s.hostmemfp16)
	//	if err != nil {
	//		return nil, err
	//	}
	//}
	//rflg := reduce.Flags
	//	rflg.IndFlag.FlattenedIndicies()
	//	rflg.IndType.Type32Bit()
	//	rflg.NanProp.Propigate()
	//	rflg.ReduceMode.MulNoZeros()
	/*
		s.op, err = reduce.Stage(rflg.ReduceMode, dtype, rflg.NanProp, rflg.IndFlag, rflg.IndType)
		if err != nil {
			return nil, err
		}
		s.reducetensor, err = layers.CreateTensor(h, y.Format(), y.DataType(), reducedims)
		if err != nil {
			return nil, err
		}
		rindsib, err := s.op.GetIndiciesSize(s.h, s.dx.Volume, s.reducetensor.Volume)
		if err != nil {
			return nil, err
		}
		s.wspacesib, err = s.op.GetWorkSpaceSize(h, s.dx.Volume, s.reducetensor.Volume)
		if err != nil {
			return nil, err
		}
		if rindsib > 0 {
			s.indicies, err = nvidia.MallocGlobal(s.h, rindsib)
			if err != nil {
				return nil, err
			}
		}
		if s.wspacesib > 0 {
			s.wspace, err = nvidia.MallocGlobal(s.h, s.wspacesib)
			if err != nil {
				return nil, err
			}

		}
	*/
	return s, nil
}

//PerformError performs softmax error
func (s *SoftMax) PerformError(x, dx, y, target *layers.Tensor) (err error) {
	//	err = s.dx.SetAll(0)
	if err != nil {
		return err
	}
	err = s.s.ForwardProp(s.h, x, y)
	if err != nil {
		return err
	}
	err = s.h.Sync()
	if err != nil {
		return err
	}
	err = dx.OpAdd(s.h, y.Volume, target.Volume, 1, -1, 0)
	if err != nil {
		return err
	}

	//s.dx.TogglePrintValueForStringer()
	//fmt.Println(s.dx)
	//s.dx.TogglePrintValueForStringer()
	//err = s.s.BackProp(s.h, dx, target, y)
	//if err != nil {
	//	return err
	//}
	//err = s.h.Sync()
	//if err != nil {
	//	return err
	//}

	//err = s.x.OpMult(s.h, s.y.Volume, s.target.Volume, 1, 1, 0)
	//if err != nil {
	//	return err
	//}
	//err = s.h.Sync()
	err = s.h.Sync()
	if err != nil {
		return err
	}
	if y == nil {
		panic("s.y is nil")
	}
	if target == nil {
		panic("target == nil")
	}
	//fmt.Println("s.h.XHandle(), s.alphaloss, s.y.TD(), s.y, s.betaloss, s.target.TD(), s.target", s.h.XHandle(), s.alphaloss, s.y.TD(), s.y, s.betaloss, s.target.TD(), s.target)
	if s.sml == nil {
		panic("s.sml is nil")
	}
	s.hostmem, err = s.sml.FindAverageLogLoss(s.h.XHandle(), s.alphaloss, y.TD(), y, s.betaloss, target.TD(), target)
	if err != nil {
		return err
	}
	//err = s.op.Reduce(s.h, s.indicies, s.wspace, 1, s.x.Volume, 0, s.reducetensor.Volume)

	err = s.h.Sync()
	if err != nil {
		return err
	}

	//err = nvidia.Memcpy(s.hostptr, s.reducetensor, s.reducetensor.SIB())

	//if err != nil {
	//	return err
	//}
	//if s.hostmemfp16 != nil {
	//err = half.FillFloat32Slice(s.hostmem, s.hostmemfp16)
	//if err != nil {
	//return err
	//}
	//for i := range s.hostmem {
	//s.hostmem[i] = -float32(math.Log10(float64(s.hostmem[i])))
	//}
	//
	//} else {
	//for i := range s.hostmem {
	//s.hostmem[i] = -float32(math.Log10(float64(s.hostmem[i])))
	//}
	//
	//}

	return nil

}

////GetTensorX returns the input tensor
//func (s *SoftMax) GetTensorX() *layers.Tensor {
//	return s.x
//}
//
////GetTensorDX returns the tensor the holds the errors for the previous layer.
//func (s *SoftMax) GetTensorDX() *layers.Tensor {
//	return s.dx
//}
//
////GetTensorY returns  the classifier output tensor
//func (s *SoftMax) GetTensorY() *layers.Tensor {
//	return s.y
//}
//
////GetTensorDY returns the tensor that holds target values.
//func (s *SoftMax) GetTensorDY() *layers.Tensor {
//	return s.target
//}
//
////SetTensorX sets the input tensor
//func (s *SoftMax) SetTensorX(x *layers.Tensor) {
//	s.x = x
//}
//
////SetTensorDX sets the tensor that holds the errors for the previous layer.
//func (s *SoftMax) SetTensorDX(dx *layers.Tensor) {
//	s.dx = dx
//}
//
////SetTensorY sets the output tensor for the classifier forward
//func (s *SoftMax) SetTensorY(y *layers.Tensor) {
//	s.y = y
//}
//
////SetTensorDY sets the tensor that holds target values.
//func (s *SoftMax) SetTensorDY(dy *layers.Tensor) {
//	s.target = dy
//}

//TestForward TestForward still calculates the AverageLoggLoss
func (s *SoftMax) TestForward(x, y, target *layers.Tensor) (err error) {
	err = s.s.ForwardProp(s.h, x, y)
	if err != nil {
		return err
	}
	s.hostmem, err = s.sml.FindAverageLogLoss(s.h.XHandle(), s.alphaloss, y.TD(), y, s.betaloss, target.TD(), target)
	return err
}

//Inference just performs the forward propagation of the classifier
func (s *SoftMax) Inference(x, y *layers.Tensor) (err error) {
	err = s.s.ForwardProp(s.h, x, y)
	if err != nil {
		return err
	}
	return nil
}

//GetAverageBatchLoss gets the averagebatchloss
func (s *SoftMax) GetAverageBatchLoss() float32 {
	return s.hostmem
	//var loss float32
	//
	//for i := range s.hostmem {
	//	loss = loss + s.hostmem[i]
	//}
	//
	//return loss / (float32)(len(s.hostmem))

}

////GetBatchLoss gets the loss by batch
//func (s *SoftMax) GetBatchLoss() []float32 {
//
//	for i := range s.hostmem {
//		s.hostmem[i] = -float32(math.Log10(float64(s.hostmem[i])))
//	}
//	return s.hostmem
//
//}

//MakeSoftMaxLossCalculator returns a loss calculator for softmax
func MakeSoftMaxLossCalculator() SoftMax {
	return SoftMax{}
}

//BatchLossCPU takes the actual and desired arrays in the form of i=batchindex, j=classindex actual[i*classificationsize+j]
func (s SoftMax) BatchLossCPU(actual, desired []float32, batchsize, classificationsize int) (percent, loss float32) {
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

//EpocLossCPU returns the loss epoc if BatchLoss was not calculated.
func (s SoftMax) EpocLossCPU(actual, desired [][]float32, batchsize, classificationsize int) (percent, loss float32) {

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
	if math.IsNaN(float64(batchloss)) {
		panic("reach NAN")
	}
	return percent / float32(numofbatches), batchloss / float32(numofbatches)
}
