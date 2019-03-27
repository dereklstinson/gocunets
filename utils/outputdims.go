package utils

import (
	"errors"
	"fmt"
)

type convprop struct {
	input, weight, output int32

	s, p, d int32
}
type ConvolutionCombo struct {
	Output                []int32
	Stride, Pad, Dilation []int32
}

//FindAllCombos will set hard limits on minpad,maxpad, maxstride,minstride,mindilation and maxdilation passed.  If those values are < 0. Then default will be set.
//That being said maxstride/minstride, and maxdilation/mindilation can't be 0.
//The min of those is 1. Maxpad can only be w-1. if you choose a value that is beyond the scope of what I have as the limits. and choose the max or min defaults.
func FindAllCombos(x, w []int32, NCHW bool, minpadding, maxpadding, minstrides, maxstrides, mindil, maxdil int32) (ccs []ConvolutionCombo) {

	wx, ww, wo := findworkingvalues(x, w, NCHW)
	xwcombs := make([]xwcombo, 0)
	for i := range wx {
		var flag bool
		td, wd := wx[i], ww[i]
		if len(xwcombs) > 0 {
			for j := range xwcombs {
				if xwcombs[j].check(td, wd) {
					flag = true
				}
			}
		}
		if !flag {
			xwcomb := createxwcombo(td, wd)
			minp := minpad(wd, minpadding)
			maxp := maxpad(wd, maxpadding)
			for j := minp; j <= maxp; j++ {
				mind := mindilation(td, wd, j, mindil)
				maxd := maxdilation(td, wd, j, maxdil)
				for k := mind; k <= maxd; k++ {
					mins := minstride(td, wd, j, k, minstrides)
					maxs := maxstride(td, wd, j, k, maxstrides)
					for l := mins; l <= maxs; l++ {
						val := findoutputdim(td, wd, l, j, k)
						if val != -1 {
							xwcomb.append([]int32{j, k, l}, 0)
						}
					}
				}
			}
			xwcombs = append(xwcombs, xwcomb)
		}
	}
	dimhascombo := make([]int32, len(wx))
	for i := range wx {
		td, wd := wx[i], ww[i]
		for j := range xwcombs {
			if xwcombs[j].check(td, wd) {
				dimhascombo[i] = int32(j)
			}
		}
	}
	fmt.Println(len(xwcombs), dimhascombo)

	comboperdim := make([]xwcombo, len(dimhascombo))
	for i := range dimhascombo {
		comboperdim[i] = xwcombs[dimhascombo[i]]
	}
	ccs = make([]ConvolutionCombo, 0)

	stride := make([]int32, len(wx))
	padding := make([]int32, len(wx))
	dilation := make([]int32, len(wx))
	ccs = getcombos(comboperdim, wo, stride, padding, dilation, ccs)
	for i := range ccs {
		ccs[i].Output = findoutputfromworkingvalues(ccs[i].Output, x, w, NCHW)
	}

	return ccs
}

type xwcombo struct {
	x, w      int32
	pds       [][]int32
	minoutval int32
}

func (xc *xwcombo) check(x, w int32) bool {
	if xc.w == w && xc.x == x {
		return true
	}
	return false
}

func FindMinOutputs(x, w []int32, NCHW bool, minpadding, maxpadding, minstrides, maxstrides, mindil, maxdil int32) (ccs []ConvolutionCombo, err error) {
	wx, ww, wo := findworkingvalues(x, w, NCHW)

	xwcombs := make([]xwcombo, 0)

	for i := range wx {
		var flag bool
		td, wd := wx[i], ww[i]
		if len(xwcombs) > 0 {
			for j := range xwcombs {

				if xwcombs[j].check(td, wd) {
					flag = true
				}
			}
		}

		if !flag {

			xwcomb := createxwcombo(td, wd)
			minval := int32(99999999)
			minp := minpad(wd, minpadding)
			maxp := maxpad(wd, maxpadding)
			for j := minp; j <= maxp; j++ {
				mind := mindilation(td, wd, j, mindil)
				maxd := maxdilation(td, wd, j, maxdil)
				for k := mind; k <= maxd; k++ {
					mins := minstride(td, wd, j, k, minstrides)
					maxs := maxstride(td, wd, j, k, maxstrides)
					for l := mins; l <= maxs; l++ {

						val := findoutputdim(td, wd, l, j, k)
						if val != -1 && val <= minval {
							minval = val
							xwcomb.append([]int32{j, k, l}, minval)

						}
					}
				}

			}
			xwcomb.makeallmin()
			xwcombs = append(xwcombs, xwcomb)
		}
	}

	dimhascombo := make([]int32, len(wx))
	for i := range wx {
		td, wd := wx[i], ww[i]
		for j := range xwcombs {
			if xwcombs[j].check(td, wd) {
				dimhascombo[i] = int32(j)
			}
		}
	}
	fmt.Println(len(xwcombs), dimhascombo)

	comboperdim := make([]xwcombo, len(dimhascombo))
	for i := range dimhascombo {
		comboperdim[i] = xwcombs[dimhascombo[i]]
	}
	ccs = make([]ConvolutionCombo, 0)

	stride := make([]int32, len(wx))
	padding := make([]int32, len(wx))
	dilation := make([]int32, len(wx))
	ccs = getcombos(comboperdim, wo, stride, padding, dilation, ccs)
	for i := range ccs {
		ccs[i].Output = findoutputfromworkingvalues(ccs[i].Output, x, w, NCHW)
	}

	return ccs, nil

}
func creatconvolutioncomb(output, pad, dilation, stride []int32) ConvolutionCombo {
	var c ConvolutionCombo
	c.Output = make([]int32, len(output))
	c.Pad = make([]int32, len(pad))
	c.Dilation = make([]int32, len(dilation))
	c.Stride = make([]int32, len(stride))
	copy(c.Output, output)
	copy(c.Pad, pad)
	copy(c.Dilation, dilation)
	copy(c.Stride, stride)
	return c
}
func getcombos(c []xwcombo, output, pad, dilation, stride []int32, allcombs []ConvolutionCombo) []ConvolutionCombo {
	slot := len(pad) - len(c)
	for _, pds := range c[0].pds {
		x := c[0].x
		w := c[0].w
		p := pds[0]
		d := pds[1]
		s := pds[2]
		pad[slot] = p
		dilation[slot] = d
		stride[slot] = s
		output[slot] = findoutputdim(x, w, s, p, d)
		if len(c) == 1 {
			allcombs = append(allcombs, creatconvolutioncomb(output, pad, dilation, stride))

		} else {
			allcombs = getcombos(c[1:], output, pad, dilation, stride, allcombs)
		}
	}
	return allcombs

}
func createxwcombo(x, w int32) xwcombo {
	return xwcombo{
		x:   x,
		w:   w,
		pds: make([][]int32, 0),
	}
}
func (xc *xwcombo) append(pds []int32, minoutval int32) {
	xc.minoutval = minoutval
	xc.pds = append(xc.pds, pds)

}
func (xc *xwcombo) makeallmin() {

	var val int32
	newpds := make([][]int32, 0)
	for _, pds := range xc.pds {
		x := xc.x
		w := xc.w
		p := pds[0]
		d := pds[1]
		s := pds[2]
		val = findoutputdim(x, w, s, p, d)
		if val == xc.minoutval {
			newpds = append(newpds, pds)
		}
	}
	xc.pds = newpds
}
func (c *ConvolutionCombo) OutputVol() int32 {
	return FindVolumeInt32(c.Output, nil)
}
func FindMaxOutput(x, w []int32, NCHW bool) (cc ConvolutionCombo, err error) {
	wx, ww, wo := findworkingvalues(x, w, NCHW)
	dilation := make([]int32, len(wx))
	padding := make([]int32, len(wx))
	stride := make([]int32, len(wx))
	for i := range wx {
		mp := maxpad(ww[i], ww[i]-1)
		padding[i] = mp
		dilation[i] = 1
		stride[i] = 1
		maxdim := findoutputdim(wx[i], ww[i], 1, mp, 1)
		if maxdim < 1 {
			return cc, errors.New("No Max Combo made negative output dim")
		}
		wo[i] = maxdim
	}
	output := findoutputfromworkingvalues(wo, x, w, NCHW)
	cc = ConvolutionCombo{
		Dilation: dilation,
		Stride:   stride,
		Pad:      padding,
		Output:   output,
	}
	return cc, nil
}
func findworkingvalues(x, w []int32, NCHW bool) (wx, ww, wo []int32) {
	convoptions := len(x) - 2
	wx = make([]int32, convoptions)
	ww = make([]int32, convoptions)
	wo = make([]int32, convoptions)
	if NCHW {

		for i := range wx {
			wx[i] = x[2+i]
			ww[i] = w[2+i]
		}
	} else {

		for i := range wx {
			wx[i] = x[1+i]
			ww[i] = w[1+i]
		}
	}
	return wx, ww, wo
}
func findoutputfromworkingvalues(workingoutput, x, w []int32, NCHW bool) (output []int32) {
	output = make([]int32, len(x))
	output[0] = x[0]
	if NCHW {
		output[1] = w[0]
		for i := range workingoutput {
			output[2+i] = workingoutput[i]
		}
	} else {
		output[len(x)-1] = w[0]
		for i := range workingoutput {
			output[1+i] = workingoutput[i]
		}
	}
	return output
}

func findoutputdim(x, w, s, p, d int32) int32 {
	y := x + (2 * p) - (((w - 1) * d) + 1)
	if y < 0 {
		return -1
	}
	return divideup(y, s) + 1
}
func minpad(w, limit int32) int32 {
	if limit < 0 || limit > (w-1) {
		return 0
	}
	return w - 1
}
func maxpad(w, limit int32) int32 {
	if limit > w-1 || limit < 0 {
		return w - 1
	}
	return limit
}
func minstride(x, w, p, d, limit int32) int32 {
	val := x + (2 * p) - (((w - 1) * d) + 1)
	if val == 0 {
		return 1
	}
	if val < 0 {
		return -1
	}

	if limit < 1 || limit > val {
		return 1
	}
	return limit
}

//maxstride can't be larger than the weights itself(self imposed)
func maxstride(x, w, p, d, limit int32) int32 {

	val := x + (2 * p) - (((w - 1) * d) + 1)
	if val == 0 {
		return 1
	}
	if val < 0 {
		return -1
	}
	if limit > val || limit < 1 {
		if val < w {
			return val
		}
		return w

	}
	return limit
}

func mindilation(x, w, p, limit int32) int32 {

	val := (x + (2 * p) - 1) / (w - 1)
	if limit < 0 || limit > val {
		return 1
	} else if limit < w {
		return limit
	}
	return w
}

//max dilation can't be larger than weights(self imposed)
func maxdilation(x, w, p, limit int32) int32 {
	val := (x + (2 * p) - 1) / (w - 1)
	if limit < 0 || limit > val {
		if val < w {
			return val
		}
		return w
	}
	return limit
}
func padcheck(w, p int32) bool {

	if p < 0 || p > w-1 {
		return false
	}
	return true
}
func stridecheck(w, s int32) bool {
	if s < 1 || s > w {
		return false
	}
	return true
}
func dilationcheck(x, w, d int32) bool {
	if d < 1 || x < d*(w-1) {
		return false
	}
	return true
}
func divideup(num, den int32) int32 {
	if num%den != 0 {
		return (num / den) + 1
	}
	return num / den
}
