package utils

import (
	"errors"
	"fmt"
)

type ConvolutionCombo struct {
	Output                []int32
	Stride, Pad, Dilation []int32
}

type dimpathxtoy struct {
	x, y int32
	path []int32
}
type dimoflayer struct {
	parent   *dimoflayer
	input    int32
	s, p, d  int32
	children []*dimoflayer
}

/*
//FindReasonAbleCombosForPath not operational
func FindReasonAbleCombosForPath(x, y []int32, layers [][]int32, NCHW bool, minpadding, minstrides, maxdil int32) (ccs []ConvolutionCombo, err error) {
	wx, wy, ww := findworkingpathvalues(x, y, layers, NCHW)
	dimpaths := makedimpaths(wx, wy, ww)
	for _, dim := range dimpaths {
		dimx := dim.x
		dimy := dim.y
		aasdasdfasdfasdfasdf := dimx + dimy
		fmt.Println(aasdasdfasdfasdfasdf)

	}
	return nil, errors.New("Not running yet")
}
*/
func reasonablepath(x, y, minp, maxs, maxd int32, w []int32) (out, p, s, d int32) {
	if y > x {

	}
	return -1, -1, -1, -1
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
