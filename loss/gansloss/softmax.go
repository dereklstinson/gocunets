package gansloss

import "math"

type SoftMax struct {
}

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
