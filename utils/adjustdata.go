package utils

//DivideAll will divide all the values in data by value
func DivideAll(data []float32, value float32) {
	for i := range data {
		data[i] /= value
	}
}

//MultiplyAll adds value to all the elements in data
func MultiplyAll(data []float32, value float32) {
	for i := range data {
		data[i] *= value
	}
}

//AddAll adds value to all the elements in data
func AddAll(data []float32, value float32) {
	for i := range data {
		data[i] += value
	}
}

//SetAll sets the data elements to value
func SetAll(data []float32, value float32) {
	for i := range data {
		data[i] = value
	}
}
