package gansloss

import (
	"math/rand"
)

type SemiSuper struct {
}

//Loss return the gLoss and dLoss for a softmax output.  Will have to pass a loss function that is wanted to use in order to use it
func (s SemiSuper) Loss(dReal, dFake []float32, batchsize, classsize int, ganlossfunc func(areal, afake []float32) (gloss, dloss float32)) (gLoss, dLoss float32) {
	batchsizeReal := len(dReal) / classsize
	batchsizeFake := len(dFake) / classsize
	RealProb := make([]float32, batchsizeReal)
	FakeProb := make([]float32, batchsizeFake)
	for i := 0; i < batchsizeReal; i++ {
		RealProb[i] = s.findrealprob(dReal[i*classsize : (i+1)*classsize])

	}
	for i := 0; i < batchsizeFake; i++ {
		FakeProb[i] = s.findrealprob(dFake[i*classsize : (i+1)*classsize])

	}
	gLoss, dLoss = ganlossfunc(RealProb, FakeProb)
	return gLoss, dLoss
}

//This only works if fake class is tacked onto the end of array
func (s SemiSuper) findrealprob(softmaxoutput []float32) float32 {
	return 1 - softmaxoutput[len(softmaxoutput)-1] // (1-fakeprob)=realprob
}
func (s SemiSuper) findbinaryanswers(real, fake []float32, classificationsize int) (binaryreal, binaryfake []float32) {
	size := len(real)
	totalbatches := size / classificationsize
	binaryreal = make([]float32, totalbatches)
	binaryfake = make([]float32, totalbatches)
	for i := 0; i < totalbatches; i++ {
		binaryreal[i] = s.findrealprob(real[i*classificationsize : (i+1)*classificationsize])
		binaryfake[i] = s.findrealprob(fake[i*classificationsize : (i+1)*classificationsize])
	}
	return binaryreal, binaryfake
}
func (s SemiSuper) findgeneratorbinary(fake []float32, classificationsize int) (binaryfake []float32) {
	size := len(fake)
	totalbatches := size / classificationsize
	binaryfake = make([]float32, totalbatches)
	for i := 0; i < totalbatches; i++ {

		binaryfake[i] = s.findrealprob(fake[i*classificationsize : (i+1)*classificationsize])
	}
	return binaryfake
}

//GeneratorBinaryError will generate a binary error just for the generator
func (s SemiSuper) GeneratorBinaryError(fake []float32, classificationsize int, smoothed bool) float32 {
	// dl/dy = -1/n * summation_1->n( desired/actual - (1-desired)/(1-actual)
	fakeb := s.findgeneratorbinary(fake, classificationsize)
	batches := len(fakeb)
	y := make([]float32, batches)
	if smoothed == true {
		for i := range y {
			y[i] = (rand.Float32() / float32(2)) + .7
		}
	}

	var summer float32
	for i := 0; i < batches; i++ {
		summer += (y[i] / fakeb[i]) - ((1 - y[i]) / (1 - fakeb[i]))
	}
	return -summer / float32(batches)
}
