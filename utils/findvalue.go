package utils

//FindMin will return the min value in array
func FindMin(data []float32) float32 {
	var min float32
	min = float32(99999999)
	for i := 0; i < len(data); i++ {
		if data[i] < min {
			min = data[i]
		}
	}
	return min
}

//FindMax will return the max value of an array
func FindMax(data []float32) float32 {
	var max float32
	max = float32(-99999999)
	for i := 0; i < len(data); i++ {
		if data[i] > max {
			max = data[i]
		}
	}
	return max
}

//FindAvg will find the average value of an array
func FindAvg(data []float32) float32 {
	avg := float32(0)
	for i := range data {
		avg += data[i]
	}
	return avg / float32(len(data))
}

func FindTotal(data []float32) float32 {
	ttl := float32(0)
	for i := range data {
		ttl += data[i]
	}
	return ttl
}
