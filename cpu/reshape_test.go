package cpu_test

import (
	"fmt"
	"testing"

	"github.com/dereklstinson/GoCuNets/cpu"
)

func TestReshape(t *testing.T) {
	dims := []int32{1, 3, 16, 16}
	tensor := helpertensor(dims)
	newvals, newdims, err := cpu.SegmentBatch1CHWtoNCHW4d(tensor, dims, 5, 5)
	if err != nil {
		t.Error(err)
	}
	toprint1 := sparator(dims, tensor)
	toprint2 := sparator(newdims, newvals)
	sectionalprint(toprint1)
	sectionalprint(toprint2)
	t.Error("YO")
}
func sparator(dims []int32, values []float32) [][][][]float32 {
	tensor := make([][][][]float32, dims[0])
	outsidecounter := 0
	zero := int32(0)
	for i := zero; i < dims[0]; i++ {
		tensor[i] = make([][][]float32, dims[1])
		for j := zero; j < dims[1]; j++ {
			tensor[i][j] = make([][]float32, dims[2])
			for k := zero; k < dims[2]; k++ {
				tensor[i][j][k] = make([]float32, dims[3])
				for l := zero; l < dims[3]; l++ {
					tensor[i][j][k][l] = values[outsidecounter]
					outsidecounter++
				}
			}
		}
	}
	return tensor
}

/*
func sectionprintbatches{
	for i := 0; i < len(input); i++ {
		for j := 0; j < len(input[i]); j++ {
			for k := 0; k < len(input[i][j]); k++ {
				for l := 0; l < len(input[i][j][k]); l++ {
					fmt.Printf("%-6.0f ", input[i][j][k][l])
				}
				fmt.Printf(" | ")
			}
			fmt.Printf("\n")
		}
		fmt.Printf("\n")
	}


}
*/
func sectionalprint(input [][][][]float32) {
	for i := 0; i < len(input); i++ {
		for j := 0; j < len(input[i]); j++ {
			for k := 0; k < len(input[i][j]); k++ {
				for l := 0; l < len(input[i][j][k]); l++ {
					fmt.Printf("%-6.0f ", input[i][j][k][l])
				}
				fmt.Printf("\n")
			}
			fmt.Printf("\n")
		}
		fmt.Printf("\n")
	}

}

func helpertensor(dims []int32) []float32 {
	zero := int32(0)
	size := cpu.Volume(dims)
	tensor := make([]float32, size)
	memspace := make([]int32, len(dims)-1)
	for i := 0; i < len(dims)-1; i++ {
		memspace[i] = cpu.Volume(dims[i+1:])
	}
	for i := zero; i < dims[0]; i++ {

		for j := zero; j < dims[1]; j++ {

			for k := zero; k < dims[2]; k++ {

				for l := zero; l < dims[3]; l++ {
					tensor[i*memspace[0]+j*memspace[1]+k*memspace[2]+l] = float32(i*memspace[0] + j*memspace[1] + k*memspace[2] + l)
				}

			}
		}
	}
	return tensor
}
