package cnndots

import (
	"fmt"

	"github.com/dereklstinson/GoCuNets/arrays"
)

/*
type HLayer3d struct {
	neurons   []arrays.HArray3d
	gradadds  []arrays.HArray3d
	biases    []float32
	bgradadds []float32
	x, y      dependents
	dropped   bool
}
type dependents struct {
	slide int
	pad   int
}
*/

func Testing() {
	var testneu HLayer3d
	testneu.neurons = make([]arrays.HArray3d, 2)
	testneu.gradadds = make([]arrays.HArray3d, 2)
	testneu.biases = make([]float32, 2)
	testneu.bgradadds = make([]float32, 2)
	testneu.x.pad = 2
	testneu.x.slide = 1
	testneu.y.pad = 2
	testneu.y.slide = 1

	testneu.neurons[0] = arrays.Harray3dTestArray2()
	testneu.neurons[1] = arrays.Harray3dTestArray3()
	testneu.gradadds[0] = testneu.neurons[0].CloneEmpty()
	testneu.gradadds[1] = testneu.neurons[0].CloneEmpty()
	testneu.biases[0] = 2
	testneu.biases[1] = 1
	output := arrays.NewHArray3d(outputcalc(4, 3, 2, 1), outputcalc(4, 3, 2, 1), len(testneu.neurons))
	input := arrays.Harray3dTestArray1()
	returngrads := input.CloneEmpty()
	fmt.Println("OUTPUT:", output)
	fmt.Println("INTPUT:", input)
	fmt.Println("LAYER", testneu)
	fmt.Println("")
	testneu.Forward(&input, &output)
	fmt.Println("")
	fmt.Println("OUTPUT:", output)
	fmt.Println("INTPUT:", input)
	fmt.Println("LAYER", testneu)
	fmt.Println("")
	output.SetAll(.5)
	fmt.Println("")
	fmt.Println("GRADS:", output)
	fmt.Println("INTPUT:", input)
	fmt.Println("RETURNGRADS", testneu)
	fmt.Println("LAYER", testneu)
	fmt.Println("")
	testneu.Backward(&input, &output, &returngrads)
	fmt.Println("")
	fmt.Println("OUTPUT:", output)
	fmt.Println("INTPUT:", input)
	fmt.Println("RETURNGRADS", returngrads)
	fmt.Println("LAYER", testneu)

	/*input := arrays.Harray3dTestArray1()
	kernel1 := arrays.Harray3dTestArray2()
	kernel2 := arrays.Harray3dTestArray3()
	gadd1 := kernel1.CloneEmpty()
	gadd2 := kernel1.CloneEmpty()
	var testneuron = HLayer3d{
		neurons:   {kernel1, kernel2},
		gradadds:  {gadd1, gadd2},
		biases:    {2, 2},
		bgradadds: {0, 0},
		x:         {1, 2},
		y:         {1, 2},
		dropped:   false,
	}
	*/
}
