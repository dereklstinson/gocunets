package dimlayer

import (
	"errors"
	"fmt"
)

type DimLayer struct {
	w       int32
	p, d, s int32
}

func CreateDimLayer(w, p, d, s int32) DimLayer {

	return DimLayer{
		w: w,
		p: p,
		d: d,
		s: s,
	}
}
func CopyDimsLayer(src []DimLayer) (copy []DimLayer) {
	copy = make([]DimLayer, len(src))
	for i := range src {
		copy[i] = CreateDimLayer(src[i].Get())
	}
	return copy
}

//Set sets the convolution values
func (l *DimLayer) Set(p, d, s int32) {

	l.p, l.d, l.s = p, d, s
}

//Get returns the prevously set values
func (l *DimLayer) Get() (w, p, d, s int32) {
	w, p, d, s = l.w, l.p, l.d, l.s
	return w, p, d, s
}

//Out returns the output from the input considering the previously the previously set values
func (l *DimLayer) Out(x int32) (y int32) {
	return findoutputdim(x, l.w, l.s, l.p, l.d)
}

//ReverseOut finds the output for the reverse convolution.
func (l *DimLayer) ReverseOut(y int32) (x int32) {
	return findreverseoutputdim(y, l.w, l.s, l.p, l.d)
}
func findoutputdim(x, w, s, p, d int32) int32 {
	y := x + (2 * p) - (((w - 1) * d) + 1)
	if y < 0 {
		return -1
	}
	return divideup(y, s) + 1
}

//MaxOutput will change the values of layers to achieve the max output for all the layers
//It will also return the value of the final layer
func MaxOutput(input int32, layers []DimLayer) (output int32) {
	output = input
	for i := range layers {
		p := layers[i].w - 1
		layers[i].Set(p, 1, 1)
		output = layers[i].Out(output)
	}
	return output
}

/*
func withinboundsy(index, y, startx, endy int32) bool {
	if y == (startx-endy-1)/(index+1) {
		return true
	}
	return false
}
*/
func withinboundsx(numoflayers, index, x, startx, endy int32) bool {
	boundx := ((startx) / (numoflayers)) * (index + 1)
	if x >= boundx {
		if index != 0 {
			if x <= boundx+2 {
				return true
			}

		} else {
			if boundx <= startx {
				return true
			}

		}
		return false
	}
	return false
}
func Backwardspdv2(xgoal, ygoal int32, dlayer []*Ofhlpr) (int32, error) {
	output := []int32{ygoal}

	var err error

	for i := len(dlayer) - 1; i >= 0; i-- {
		output, err = dlayer[i].backwardmultinputs(output)
		for i := range output {
			fmt.Println(output[i])
		}
		if err != nil {
			return -1, err
		}
	}
	for i := range output {
		fmt.Println(output[i])
		if output[i] == xgoal {
			return xgoal, nil
		}
	}

	return -1, errors.New("backwardspdv2 Didn't work")
}

func Makeoutputfinderhelper(index, totaldimlayers, globalxgoal, globalygoal, mins, mind, minp, maxs, maxd, maxp int32, layer *DimLayer) *Ofhlpr {
	fwd := indexes{s: maxs, d: maxd, p: maxp}
	bwd := indexes{s: mins, d: mind, p: minp}
	min := indexes{s: mins, d: mind, p: minp}
	max := indexes{s: maxs, d: maxd, p: maxp}

	return &Ofhlpr{
		index:          index,
		totaldimlayers: totaldimlayers,
		gbyg:           globalygoal,
		gbxg:           globalxgoal,
		max:            max,
		min:            min,
		layer:          layer,
		fwd:            fwd,
		bwd:            bwd,
	}
}
func (h *Ofhlpr) backwardmultinputs(yinputs []int32) (outputs []int32, err error) {

	for i := range yinputs {
		h.backwardspd(yinputs[i])
	}
	outputarray := h.getreverseoutputsvals()
	if len(outputarray) < 1 || outputarray == nil {
		return outputarray, fmt.Errorf("nothing found")
	}
	return outputarray, nil
}
func (h *Ofhlpr) backwardspd(yinput int32) (output []int32, err error) {

	for ; h.bwd.s <= h.max.s; h.bwd.s++ {
		for ; h.bwd.d <= h.max.d; h.bwd.d++ {
			for ; h.bwd.p <= h.max.p; h.bwd.p++ {
				output := findreverseoutputdim(yinput, h.layer.w, h.bwd.s, h.bwd.p, h.bwd.d)
				if withinboundsx(h.totaldimlayers, h.index, output, h.gbxg, h.gbyg) {
					fmt.Println(output)
					h.append(yinput, output, h.bwd.s, h.bwd.d, h.bwd.d)
				}

			}
		}
	}
	outputarray := h.getreverseoutputsvals()
	if len(outputarray) < 1 || outputarray == nil {
		return outputarray, fmt.Errorf("nothing found")
	}
	return outputarray, nil
}
func (h *Ofhlpr) getreverseoutputsvals() []int32 {
	outputs := make([]int32, len(h.outputs))
	for i := range h.outputs {
		outputs[i] = h.outputs[i].output
	}
	return outputs
}

type Ofhlpr struct {
	gbxg, gbyg, index, totaldimlayers int32
	max                               indexes
	min                               indexes
	bwd                               indexes
	fwd                               indexes
	layer                             *DimLayer
	outputs                           []routputs
}

func compairpdswithindex(a indexes, s, d, p int32) bool {
	if a.s == s && a.p == p && a.d == d {
		return true
	}
	return false
}
func compairindexes(a, b indexes) bool {
	if a.s == b.s && a.p == b.p && a.d == b.d {
		return true
	}
	return false
}
func makeroutput(output, input, s, d, p int32) routputs {
	inputs := make([]rinput, 1)
	inputs[0] = makeinput(input, s, d, p)
	return routputs{
		output: output,
		inputs: inputs,
	}
}
func makeinput(input, s, d, p int32) rinput {
	combo := make([]indexes, 1)
	combo[0] = indexes{s: s, d: d, p: p}
	return rinput{
		input:  input,
		combos: combo,
	}
}

type rinput struct {
	input  int32
	combos []indexes
}

func (r *routputs) append(input, s, d, p int32) {
	var hasinput bool
	if len(r.inputs) < 1 || r.inputs == nil {
		r.inputs = make([]rinput, 0)
	}
	for i := range r.inputs {
		if r.inputs[i].input == input {
			hasinput = true
			r.inputs[i].append(s, d, p)
			break
		}
	}
	if !hasinput {
		r.inputs = append(r.inputs, makeinput(input, s, d, p))
	}
}
func (r *rinput) append(s, d, p int32) {

	var hascombo bool
	if len(r.combos) < 1 || r.combos == nil {
		r.combos = make([]indexes, 0)
		hascombo = false

	}
	for i := range r.combos {
		if r.combos[i].s == s && r.combos[i].d == d && r.combos[i].p == p {
			hascombo = true
			break
		}
	}
	if !hascombo {
		r.combos = append(r.combos, indexes{s: s, d: d, p: p})
	}
}
func (h *Ofhlpr) append(input, output, s, d, p int32) {
	var hasoutput bool
	if h.outputs == nil || len(h.outputs) < 1 {
		h.outputs = make([]routputs, 0)
	}
	for i := range h.outputs {
		if h.outputs[i].output == output {
			hasoutput = true
			h.outputs[i].append(input, s, d, p)
			break
		}
	}
	if !hasoutput {
		h.outputs = append(h.outputs, makeroutput(output, input, s, d, p))
	}
}

type routputs struct {
	output int32
	inputs []rinput
}
type indexes struct {
	s, d, p int32
}

func divideup(num, den int32) int32 {
	if num%den != 0 {
		return (num / den) + 1
	}
	return num / den
}
func findreverseoutputdim(y, w, s, p, d int32) int32 {
	// output = 1+ ((input + (2*padding) - (((filter-1)*dilation)+1))/slide)
	// *now input<==>output
	//input=  1+ ((output + (2*padding) - (((filter-1)*dilation)+1))/slide)
	//	(input-1)*slide = (output +2*padding)-(((filter-1)*dilation)+1)
	//output= 2*padding-(((filter-1)*dilation)+1)-(input-1)*slide
	// input = 1 + (output + (2*padding) - (((filter-1)*dilation)+1))/slide
	//  slide *(input-1) = output + (2*padding) - (((filter-1)*dilation)+1)
	//  output = (slide *(input-1)) - (2*padding) + (((filter-1)*dilation)+1)
	return (s * (y - 1)) - (2 * p) + (((w - 1) * d) + 1)
}
