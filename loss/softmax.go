package loss

import "math"

//SoftMax Holds the methods to do softmax loss
type SoftMax struct {
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
