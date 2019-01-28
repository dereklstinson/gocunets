package cpu_test

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/dereklstinson/GoCuNets/cpu"
)

/*
func main(t *testing.T) {
	TestReshapeHWC(t)
}
*/
func TestReshapeCHW(t *testing.T) {
	dims := []int32{1, 3, 16, 16}
	tensor := helpertensor(dims)
	newvals, newdims, hwratio, err := cpu.ShapeToBatchNCHW4DForward(tensor, dims, []int32{5, 5}, []int32{5, 5})
	if err != nil {
		t.Error(err)
	}
	fmt.Println(hwratio)
	//	toprint1 := sparator(dims, tensor)
	toprint1 := sparator(newdims, newvals)
	sectionalprint(toprint1)
	zerotensor := make([]float32, len(tensor))
	err = cpu.ShapeToBatchNCHW4DBackward(zerotensor, dims, newvals, newdims, []int32{5, 5})
	for i := 0; i < len(zerotensor); i++ {
		if tensor[i] != zerotensor[i] {
			t.Error("New Tensor Doesn't match old")
		}
	}
	//newnewvals, newnewdims, err := cpu.ShapeToBatchNCHW4DBackward(newvals, newdims, 16, 16)
	if err != nil {
		t.Error(err)
	}
	toprint2 := sparator(dims, tensor)
	sectionalprint(toprint2)

}
func TestReshapeHWC(t *testing.T) {
	runtime.LockOSThread()
	dims := []int32{1, 16, 16, 3}
	tensor := helpertensor(dims)
	newvals, newdims, err := cpu.ShapeToBatchNHWC4DForward(tensor, dims, []int32{5, 5}, []int32{5, 5})
	if err != nil {
		t.Error(err)
	}
	//fmt.Println(tensor)

	//fmt.Println(newvals)

	zerotensor := make([]float32, len(tensor))
	err = cpu.ShapeToBatchNHWC4DBackward(zerotensor, dims, newvals, newdims, []int32{5, 5})
	var flag bool
	for i := 0; i < len(zerotensor); i++ {
		if tensor[i] != zerotensor[i] {
			flag = true
		}
	}
	if flag == true {
		t.Error("Tensors Don't Match Up")
	}
	if err != nil {
		t.Error(err)
	}
	//toprint2 := sparator(dims, zerotensor)
	//	fmt.Println(zerotensor)
	//fmt.Println("Printing OLD")
	//	sectionalprinthwc(toprint2)

}
func sparator(dims []int32, values []float32) [][][][]float32 {
	if len(dims) == 0 {
		panic(dims)
	}
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
func sparatorhwc(dims []int32, values []float32) [][][][]float32 {
	if len(dims) == 0 {
		panic(dims)
	}
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
					fmt.Printf("% -4.0f ", input[i][j][k][l])
				}
				fmt.Printf("\n")
			}
			fmt.Printf("\n")
		}
		fmt.Printf("\n")
	}

}
func sectionalprinthwc(input [][][][]float32) {
	for i := 0; i < len(input); i++ {
		for j := 0; j < len(input[i]); j++ {
			for k := 0; k < len(input[i][j]); k++ {
				for l := 0; l < len(input[i][j][k]); l++ {
					fmt.Printf(" % -3.0f ", input[i][j][k][l])
				}
				fmt.Printf(",")
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
