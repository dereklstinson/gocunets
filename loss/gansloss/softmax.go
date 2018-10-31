package gansloss

/*

import "math"

//Softmax Gan loss. Should be calculated (i think) with non activated outputs
type SoftMax struct {
}

//Loss is the loss of a softmax gan
func (s SoftMax) Loss(dReal, dFake []float32) (dLoss, gLoss float32) {

	z := s.zcpu(dReal, dFake)
	dLoss = s.dlosscpu(dReal, z)
	gLoss = s.glosscpu(dReal, dFake, z)
	return dLoss, gLoss
}
func (s SoftMax) dlosscpu(dReal []float32, z float32) float32 {
	dloss := float32(0)
	for i := range dReal {
		dloss += dReal[i]
	}
	return (dloss / float32(len(dReal))) + float32(math.Log(float64(z)))
}
func (s SoftMax) glosscpu(dReal, dFake []float32, z float32) float32 {
	gloss := float32(0)
	for i := range dReal {
		//	gloss += (gtarget[i] * dreal[i])+(gtarget[i]*dfake[i])
		gloss += (dReal[i] + dFake[i])
	}
	return (gloss / float32(2*len(dFake))) + float32(math.Log(float64(z)))
}

func (s SoftMax) zcpu(dreal, dfake []float32) float32 {
	r := 0.0
	//r,f:=float32(0),float32(0)
	for i := range dreal {
		r += math.Exp(float64(-dreal[i]))
		r += math.Exp(float64(-dfake[i]))
	}

	return float32(r)
}

//BatchOutput is the output for the batch
func (s SoftMax) BatchOutput(BinaryReal, BinaryFake []float32) (RealOut, FakeOut []float32) {

	xisummation := float32(0)
	RealOut = make([]float32, len(BinaryReal))
	FakeOut = make([]float32, len(BinaryFake))
	for j := range BinaryFake {
		FakeOut[j] = float32(math.Exp(-float64(BinaryFake[j])))
		xisummation += FakeOut[j]
	}
	for j := range BinaryReal {
		RealOut[j] = float32(math.Exp(-float64(BinaryReal[j])))
		xisummation += RealOut[j]
	}
	for j := range RealOut {
		RealOut[j] /= xisummation

	}
	for j := range FakeOut {
		FakeOut[j] /= xisummation

	}
	return RealOut, FakeOut
}

//BackGradients is the derivative seperated on the batches
func (s SoftMax) BackGradients(Fake, Real []float32) (FakeDerivative, RealDerivative []float32) {
	FakeTarget := float32(0)
	FakeBatch := float32(len(Fake))
	RealTarget := float32(1)
	RealBatch := float32(len(Real))
	RealDerivative = make([]float32, len(Real))
	FakeDerivative = make([]float32, len(Fake))
	for i := range Real {
		RealDerivative[i] = (RealTarget / Real[i]) / RealBatch
	}
	for i := range Fake {
		FakeDerivative[i] = ((1 - FakeTarget) / (1 - Fake[i])) / FakeBatch

	}
	return FakeDerivative, RealDerivative
}
*/
/*  I don't think this will work so I am putting it on the back burner
func (s SoftMax) dderivative(BinaryOutReal, BinaryOutFake []float32) (xjDesc, xj1Gen, xj2Gen []float32) {
	//xi:=make([]float32,0)
	//xi=append(xi, BinaryOutReal...)
	//xi=append(xi, BinaryOutFake...)
	xisummation := float32(0)
	xjDesc = make([]float32, len(BinaryOutReal))
	xj1Gen = make([]float32, len(BinaryOutReal))
	xj2Gen = make([]float32, len(BinaryOutFake))
	for j := range BinaryOutFake {
		xj2Gen[j] = float32(math.Exp(-float64(BinaryOutFake[j])))
		xisummation += xj2Gen[j]
	}
	for j := range BinaryOutReal {
		xj1Gen[j] = float32(math.Exp(-float64(BinaryOutReal[j])))
		//	xjDesc[j]=xj1Gen[j]
		xisummation += xj1Gen[j]
	}
	for j := range xj1Gen {
		xj1Gen[j] /= xisummation
		//xjDesc[j]/=xisummation
		xj1Gen[j] = BinaryOutReal[j] * (xj1Gen[j] - 1)
		xjDesc[j] = xj1Gen[j]
	}
	for j := range xj2Gen {
		xj2Gen[j] /= xisummation
		xj2Gen[j] = BinaryOutFake[j] * (xj2Gen[j] - 1)
	}
	return
}
*/
