package cnndots

import (
	"sync"

	arrays "github.com/dereklstinson/GoCuNets/OBSOLETE-ARRAYS"
)

//hdots contains host side dot product methods for harrays. Like feedforward and backprop functions. It also contains info on
//modifying info that might be had. ie padding slide

type dependents struct {
	slide int
	pad   int
}

type HLayer3d struct {
	neurons   []arrays.HArray3d
	gradadds  []arrays.HArray3d
	biases    []float32
	bgradadds []float32
	x, y      dependents
	dropped   bool
}

func outputcalc(i, f, p, s int) int {
	return (i - f + (2 * p)) / s
}

//Forward is the feedforward function of the Convolution Layer
func (layer *HLayer3d) Forward(input *arrays.HArray3d, output *arrays.HArray3d) {
	sx, sy := layer.x.slide, layer.y.slide
	outx := output.X
	outy := output.Y
	inx := input.X
	iny := input.Y
	var wg sync.WaitGroup
	for neuron := 0; neuron < len(layer.neurons); neuron++ {
		wg.Add(1)
		go func(neuron int, input arrays.HArray3d) {
			fxsize := layer.neurons[neuron].X
			fysize := layer.neurons[neuron].Y
			for ax, px := 0, -layer.x.pad; ax < outx; ax, px = ax+1, px+sx { //I am hoping the compiler will see this and do indexing
				for ay, py := 0, -layer.y.pad; ay < outy; ay, py = ay+1, py+sy { //same here
					adder := float32(0.0)
					for fx := 0; fx < fxsize; fx++ {
						ox := px + fx
						for fy := 0; fy < fysize; fy++ {
							oy := py + fy
							if ox >= 0 && ox < inx && oy >= 0 && oy < iny {
								nxy := (fx * layer.neurons[neuron].Y * layer.neurons[neuron].Z) + (fy * layer.neurons[neuron].Z)
								nd := layer.neurons[neuron].Z
								vxy := (ox * input.Y * input.Z) + (oy * input.Z)
								for fd := 0; fd < nd; fd++ {
									adder += layer.neurons[neuron].Data[nxy+fd] * input.Data[vxy+fd]
								}

							}

						}
					}
					adder += layer.biases[neuron]
					output.Insert(ax, ay, neuron, adder)
				}
			}
			wg.Done()
		}(neuron, *input)
	}
	wg.Wait()
}

//Backward is for the backprop function of the Convolution Layer
func (layer *HLayer3d) Backward(input *arrays.HArray3d, outputgrads *arrays.HArray3d, returngrads *arrays.HArray3d) {
	sx, sy := layer.x.slide, layer.y.slide
	outx := outputgrads.X
	outy := outputgrads.Y
	inx := input.X
	iny := input.Y
	inz := input.Z
	var wg sync.WaitGroup
	var mux sync.RWMutex
	//tempgrads:=make([]arrays.HArray3d,len(layer.neurons))

	for neuron := 0; neuron < len(layer.neurons); neuron++ {

		wg.Add(1)
		go func(neuron int, input *arrays.HArray3d) {
			returngradarray := make([]float32, len(input.Data))
			fxsize := layer.neurons[neuron].X
			fysize := layer.neurons[neuron].Y
			for ax, px := 0, -layer.x.pad; ax < outx; ax, px = ax+1, px+sx { //I am hoping the compiler will see this and do indexing
				for ay, py := 0, -layer.y.pad; ay < outy; ay, py = ay+1, py+sy { //same here
					grad := outputgrads.ValueAt(ax, ay, neuron) // output location which is also the gradient location
					for fx := 0; fx < fxsize; fx++ {            //f is location on filter
						ox := px + fx //o is location on input
						for fy := 0; fy < fysize; fy++ {
							oy := py + fy
							if ox >= 0 && ox < inx && oy >= 0 && oy < iny {
								//       filter location
								nxy := (fx * layer.neurons[neuron].Y * layer.neurons[neuron].Z) + (fy * layer.neurons[neuron].Z)
								// neuron depth location
								nd := layer.neurons[neuron].Z
								// inx is inputsize x ... iny is inputsize y so vxy is the inputlocation
								vxy := (ox * iny * inz) + (oy * inz)
								for fd := 0; fd < nd; fd++ {
									idx1 := vxy + fd //
									idx2 := nxy + fd
									layer.gradadds[neuron].Data[idx2] += input.Data[idx1] * grad
									//	returngrads.Data[idx1] += layer.neurons[neuron].Data[idx2] * grad
									returngradarray[idx1] += layer.neurons[neuron].Data[idx2] * grad
								}

							}

						}
					}
					layer.bgradadds[neuron] += grad

				}
			}
			mux.Lock()
			for i := 0; i < len(returngradarray); i++ {
				returngrads.Data[i] += returngradarray[i]
			}
			mux.Unlock()
			wg.Done()
		}(neuron, input)
	}
	wg.Wait()
}
