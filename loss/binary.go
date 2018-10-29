package loss

import "math"

//Binary Struct holds the binary loss amd derovatove calculations
type Binary struct {
}

// MakeBinaryCalculator returns a Binary struct used to calculate binary stuff
func MakeBinaryCalculator() Binary {
	return Binary{}
}

func (b Binary) loss32(target, predicted float32) float32 {
	t, p := float64(target), float64(predicted)
	return -float32((t * math.Log(p)) + ((1 - t) * math.Log(1-p)))
}

//LossN returns the loss of N batches of binary outputs
func (b Binary) LossN(target, predicted []float32) float32 {
	sum := float32(0)
	for i := 0; i < len(target); i++ {
		sum += b.loss32(target[i], predicted[i])

	}
	return sum / float32(len(target))
}
func (b Binary) loss64(target, predicted float64) float64 {
	return -((target * math.Log(predicted)) + ((1 - target) * math.Log(1-predicted)))
}

//DerivativeNBatched returns the binary derivative for N batches
func (b Binary) DerivativeNBatched(target, predicted []float32) (allsame []float32) {
	allsame = make([]float32, len(target))
	batches := float32(len(target))
	sum := float32(0)
	for i := 0; i < len(target); i++ {
		sum += (target[i] / predicted[i]) - ((1 - target[i]) / (1 - predicted[i]))
	}

	for i := range allsame {
		allsame[i] = sum / batches
	}
	return allsame
}

//DerivativeNSeperated is the derivative seperated on the batches
func (b Binary) DerivativeNSeperated(target, predicted []float32) (alldifferent []float32) {
	alldifferent = make([]float32, len(target))
	for i := 0; i < len(target); i++ {
		alldifferent[i] = (target[i] / predicted[i]) - ((1 - target[i]) / (1 - predicted[i]))
	}
	return alldifferent
}
