package main

import (
	"fmt"

	"github.com/dereklstinson/GoCuNets/utils/outputdim"
)

func main() {
	y := []int32{10, 10, 1, 1}
	xs := [][]int32{
		{10, 3, 32, 32},
		//	{10, 3, 64, 64},
		//	{10, 3, 128, 128},
	}
	layers := [][]int32{
		{20, 3, 5, 5},
		{20, 20, 5, 5},
		{20, 20, 5, 5},
		{20, 20, 5, 5},
		{20, 20, 5, 5},
		{20, 20, 5, 5},
		{20, 20, 5, 5},
		{20, 20, 5, 5},
	}
	for i := range xs {
		vals, err := outputdim.FindReasonAbleCombosForPath(xs[i], y, layers, true, 2, -1, -1, true)

		fmt.Println(vals)
		if err != nil {
			panic(err)
		}
	}
}
