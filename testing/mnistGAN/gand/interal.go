package gand

import "fmt"

func dims(args ...int) []int32 {

	length := len(args)
	x := make([]int32, length)
	for i := 0; i < length; i++ {
		x[i] = int32(args[i])
	}
	return x
}
func cherror(input error) {
	if input != nil {
		fmt.Println("***************************")
		panic(input)

	}
}
func getsize(dims []int32) int32 {
	mult := int32(1)
	for i := range dims {
		mult *= dims[i]
	}
	return mult

}
