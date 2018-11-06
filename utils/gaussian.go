package utils

import (
	"math"
	"math/rand"
	"sort"
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

//RandomFloat32 returns a random float32
func RandomFloat32(floor, ceiling float32) float32 {

	return (rand.Float32() * (ceiling - floor)) + (floor)
}

//RandomFloat64 returns a random float64
func RandomFloat64(floor, ceiling float64) float64 {

	return (rand.Float64() * (ceiling - floor)) + (floor)
}

//NormalSortedGaussianVector sorts randomlygenerated vector of gaussian values that are then normalized
func NormalSortedGaussianVector(length int) []float32 {
	slice := make([]float32, length)
	adder := float32(0.0)
	for i := 0; i < length; i++ {
		slice[i] = float32(gauassian())
		adder += slice[i]
	}

	sort.Sort(f32slice(slice))
	for i := range slice {
		slice[i] /= adder
	}
	return slice

}

//NormalGaussianVector  randomlygenerated vector of gaussian values that are then normalized
func NormalGaussianVector(length int) []float32 {
	slice := make([]float32, length)
	adder := float32(0.0)
	for i := 0; i < length; i++ {
		slice[i] = float32(gauassian())
		adder += slice[i]
	}

	for i := range slice {
		slice[i] /= adder
	}
	return slice

}

//RandomSigmaGaussian2dKernel returns a randomly generated Gaussian 2d Kernal
func RandomSigmaGaussian2dKernel(h, w int) []float32 {
	sigma := (rand.Float32() + 1) * float32(h+w) / 2
	return Gaussian2dKernel(sigma, h, w)

}

//Gaussian2dKernel will return a
func Gaussian2dKernel(sigma float32, h, w int) []float32 {

	mean := float64(h+w) / 4.0 //crewed radius
	sigma1 := float64(sigma)
	kernel := make([]float32, h*w)
	adder := float32(0.0)
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			x := float64(i)
			y := float64(j)
			point := gaussiankern(x, mean, sigma1) * gaussiankern(y, mean, sigma1)
			kernel[i*w+j] = float32(point)
			adder += kernel[i*w+j]
		}
	}

	//now Normalize it
	for i := range kernel {
		kernel[i] /= adder
	}
	return kernel
}
func gaussiankern(x, mu, sig float64) float64 {
	a := (x - mu) / sig
	return math.Exp(-(a * a) / 2)
}

//RandomGaussianKernelsInChannels makes 2d GaussianKernels that are in the channels
func RandomGaussianKernelsInChannels(h, w, c int, hwc bool) []float32 {

	kernels := make([]float32, 0, h*w*c)
	for i := 0; i < c; i++ {
		kernels = append(kernels, RandomSigmaGaussian2dKernel(h, w)...)
	}
	if hwc == false {
		return kernels
	}
	tensor := make([]float32, h*w*c)
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			for k := 0; k < c; k++ {
				tensor[(i*w*c)+(j*c)+k] = kernels[(k*h*w)+(i*w)+j]
			}
		}
	}
	return tensor

}

type f32slice []float32

func (f f32slice) Len() int {
	return len(f)
}
func (f f32slice) Swap(i, j int) {
	f[i], f[j] = f[j], f[i]
}
func (f f32slice) Less(i, j int) bool {
	/*if f[i]<f[j]==true{
		return true
	}*/
	return f[i] < f[j]
}
func outerproductvol(v1, v2, v3 []float32) []float32 {
	output := make([]float32, len(v1)*len(v2)*len(v3))
	a := len(v2)
	b := len(v3)
	for i := 0; i < len(v1); i++ {
		for j := 0; j < len(v2); j++ {
			for k := 0; k < len(v3); k++ {
				output[(i*a*b)+(j*b)+k] = v1[i] * v2[j] * v3[k]
			}

		}
	}
	return output
}

//Gaussian returns the gaussien at zero
func gaussianstd3333() float32 {
	return float32(Gaussian(0, .3333))
}
