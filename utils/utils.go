package utils

import (
	"math"
	"math/rand"
)

//Gaussian returns the gaussien at zero
func Gaussian(mean float64, std float64) float64 {
	return mean + std*gauassian()
}

func gauassian() float64 {
	//Polar method
	var x, y, z float64
	for z >= 1 || z == 0 {
		x = (2 * rand.Float64()) - float64(1)
		y = (2 * rand.Float64()) - float64(1)
		z = x*x + y*y
	}
	return float64(x * math.Sqrt(-2*math.Log(z)/z))
}

//RandWeightSet sets a randomweight based on min max values
func RandWeightSet(mean, std, fanin float64) float64 {
	return Gaussian(mean, std) * (math.Sqrt((2.0) / (fanin)))
}
