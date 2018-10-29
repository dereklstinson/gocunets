package loss

//MSE is Mean Squared Error
type MSE struct {
}

//LossCPU is the loss for the output
func (m MSE) LossCPU(generated, target []float32) float32 {
	sumation := float32(0)
	for i := range target {
		x := generated[i] - target[i]
		sumation += (x * x)
	}
	return sumation / float32(len(target))
}

//DerivativeCPU is used for backprop
func (m MSE) DerivativeCPU(generated, target []float32) []float32 {
	der := make([]float32, len(generated))
	for i := range generated {
		der[i] = generated[i] - target[i]
	}
	return der
}
