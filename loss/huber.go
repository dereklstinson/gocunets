package loss

import "github.com/dereklstinson/GoCuNets/utils"

//Huber holds the methods to do the huber loss
type Huber struct {
}

//MakeHuberCalculator returns a Huber so that Huber calculations can be made
func MakeHuberCalculator() Huber {
	return Huber{}
}

//delta is given by user
func (h Huber) huberloss(target, predicted, delta float32) float32 {
	y := target - predicted
	x := utils.AbsoluteValue(y)
	if x < delta {
		return (y * y) / 2
	}
	return delta*y - delta/2
}
