package utils

import (
	"errors"
	"fmt"
)

type ConvolutionCombo struct {
	Output                []int32
	Stride, Pad, Dilation []int32
}
type ConvolutionSettings struct {
	WorkingWs, Pad, Dilation, Stride []int32
}

func createconvolutionsettings(workingws, pad, dilation, stride []int32) ConvolutionSettings {
	return ConvolutionSettings{
		WorkingWs: workingws,
		Pad:       pad,
		Dilation:  dilation,
		Stride:    stride,
	}
}

type dimlayer struct {
	w       int32
	p, d, s int32
}
type workinglayer struct {
	w, p, d, s []int32
}

func createworkinglayer() (w workinglayer) {

	w.w = make([]int32, 0)
	w.p = make([]int32, 0)
	w.d = make([]int32, 0)
	w.s = make([]int32, 0)
	return w
}
func (w *workinglayer) append(d dimlayer) {
	w.w = append(w.w, d.w)
	w.p = append(w.p, d.p)
	w.d = append(w.d, d.d)
	w.s = append(w.s, d.s)
}
func (w *workinglayer) get() (ww, wp, wd, ws []int32) {
	return w.w, w.p, w.d, w.s
}
func createdimlayer(w, p, d, s int32) dimlayer {

	return dimlayer{
		w: w,
		p: p,
		d: d,
		s: s,
	}
}
func copydimslayer(src []dimlayer) (copy []dimlayer) {
	copy = make([]dimlayer, len(src))
	for i := range src {
		copy[i] = createdimlayer(src[i].get())
	}
	return copy
}

func (l *dimlayer) set(p, d, s int32) {

	l.p, l.d, l.s = p, d, s
}
func (l *dimlayer) get() (w, p, d, s int32) {
	w, p, d, s = l.w, l.p, l.d, l.s
	return w, p, d, s
}
func (l *dimlayer) out(x int32) (y int32) {
	return findoutputdim(x, l.w, l.s, l.p, l.d)
}

type dimpath struct {
	x, y int32
	path []int32
}

func makedimpaths(wx, wy []int32, ww [][]int32) (dimpaths []dimpath) {
	dimpaths = make([]dimpath, len(wx))
	for i := range dimpaths {
		dimpaths[i].x = wx[i]
		dimpaths[i].y = wy[i]
		dimpaths[i].path = make([]int32, len(ww))
		for j := range ww {
			dimpaths[i].path[j] = ww[j][i]
		}
	}
	return dimpaths
}

//FindReasonAbleCombosForPath not operational
func FindReasonAbleCombosForPath(x, y []int32, layers [][]int32, NCHW bool, minp, maxs, maxd int32, favordilation bool) (css []ConvolutionSettings, err error) {
	fmt.Println("Starting")
	wx, wy, ww := findworkingpathvalues(x, y, layers, NCHW)
	dimpaths := makedimpaths(wx, wy, ww)
	workingoutputs := make([]int32, len(wx))
	dimpathlayers := make([][]dimlayer, len(wx))

	for i, dim := range dimpaths {

		var err error
		output, dpath, err := reasonablepath(dim.x, dim.y, minp, maxs, maxd, dim.path, favordilation)
		if err != nil {
			fmt.Println("Doing Recursive Reason")
			output1, dpath1, err2 := recursivenotreason(dim.x, dim.y, 0, maxs, maxd, dpath)
			//	fmt.Println(output1, dpath1)
			if err2 != nil {
				return nil, errors.New("no path found")
			}
			workingoutputs[i], dimpathlayers[i] = output1, dpath1
			//fmt.Printf("\n")
		} else {
			workingoutputs[i], dimpathlayers[i] = output, dpath
		}

	}
	wlayers := make([]workinglayer, len(layers))
	for i := range wlayers {
		wlayers[i] = createworkinglayer()
	}

	for i := range wlayers {
		for j := range dimpathlayers {
			wlayers[i].append(dimpathlayers[j][i])
		}

	}
	css = make([]ConvolutionSettings, len(layers))
	for i := range css {

		css[i] = createconvolutionsettings(wlayers[i].get())
	}
	return css, nil
}

//reasonablepath tries to find a path where all the layers share the same pad, stride and dim. The deeper the network the more likely this is to work.
//if minp is larger than the maxp in one of the layers. Then minp will be set to zero. This is to maximize the chance that this will work correctly.
//Error will return if it couldn't find  a path  that share all the same vals, but it will return the closest.  with the least amount of padding,stride,and dilation
func reasonablepath(x, y, minp, maxs, maxd int32, w []int32, favordilation bool) (output int32, dimlayers []dimlayer, err error) {
	dimlayers = make([]dimlayer, len(w))
	closestdimlayers := make([]dimlayer, len(w))
	smallestmaxp := int32(9999999)

	for i := range w {
		closestdimlayers[i] = createdimlayer(w[i], 1, 1, 1)
		dimlayers[i] = createdimlayer(w[i], 1, 1, 1)
		if w[i]-1 < smallestmaxp {
			smallestmaxp = w[i] - 1
		}
	}
	if smallestmaxp < minp {
		minp = 0
	}
	smallestmaxd := int32(99999)
	for i := range w {
		dilmax := maxdilation(x, w[i], smallestmaxp, maxd)
		if smallestmaxd > dilmax {
			smallestmaxd = dilmax
		}
	}
	smallestmaxs := int32(99999)
	for i := range w {
		stridex := maxstride(x, w[i], smallestmaxp, smallestmaxd, maxs)
		if smallestmaxs > stridex {
			smallestmaxs = stridex
		}
	}

	closestdist := int32(9999999)
	closestchecker := int32(0)
	if favordilation {
		for i := minp; i <= smallestmaxp; i++ {
			for j := int32(1); j <= smallestmaxd; j++ {
				for k := int32(1); k <= smallestmaxs; k++ {
					output = x
					for l := range dimlayers {
						dimlayers[l].set(i, j, k)
						output = dimlayers[l].out(output)
						if output < 0 {
							break
						}
					}
					if output == y {

						return output, dimlayers, nil
					}
					closestchecker = distancecheck(y, output)
					if closestchecker < closestdist {
						closestdist = closestchecker
						for l := range dimlayers {
							closestdimlayers[l].set(i, j, k)
							output = dimlayers[l].out(output)
						}

					}

				}
			}
		}
	} else {
		for i := minp; i <= smallestmaxp; i++ {
			for j := int32(1); j <= smallestmaxs; j++ {
				for k := int32(1); k <= smallestmaxd; k++ {
					output = x
					for l := range dimlayers {
						dimlayers[l].set(i, k, j)
						output = dimlayers[l].out(output)
						if output < 0 {
							break
						}
					}
					if output == y {
						return output, dimlayers, nil
					}
					closestchecker = distancecheck(y, output)
					if closestchecker < closestdist {
						closestdist = closestchecker
						for l := range dimlayers {
							closestdimlayers[l].set(i, k, j)
							output = dimlayers[l].out(output)
						}

					}
				}
			}
		}
	}
	dimlayers = closestdimlayers
	return output, dimlayers, errors.New("Didn't Find ReturningClosest")
}

//mostreasonableunreasonable I think I am going to get rid of it.  I like the recursive one better.  This one only changes one layer as it goes down the layer
func mostreasonableunreasonable(x, y int32, reasonable []dimlayer) (output int32, dimslayer []dimlayer, err error) {
	dimslayer = copydimslayer(reasonable)
	closestdimlayers := make([]dimlayer, len(reasonable))

	closestdist := int32(9999999)
	closestchecker := int32(0)
	for h := range dimslayer {
		dimslayer = copydimslayer(reasonable)
		w := dimslayer[h].w
		maxp := maxpad(w, -1)
		maxd := maxdilation(x, w, maxp, -1)
		maxs := maxstride(x, w, maxp, maxd, -1)

		for i := int32(0); i <= maxp; i++ {
			for j := int32(1); j <= maxd; j++ {
				for k := int32(1); k <= maxs; k++ {
					dimslayer[h].set(i, j, k)
					output = dimslayer[h].out(x)
					if output < 0 {
						break
					}
					for l := range dimslayer[h+1:] {
						output = dimslayer[l].out(output)

						if output < 0 {
							break
						}
					}
					if output > 0 {
						if output == y {

							return output, dimslayer, nil
						}
						closestchecker = distancecheck(y, output)
						if closestchecker < closestdist {
							closestdist = closestchecker
							for l := range dimslayer {
								closestdimlayers[l].set(i, j, k)
								output = dimslayer[l].out(output)
							}

						}
					}

				}
			}
		}
	}
	return output, closestdimlayers, errors.New("Couldn't find anything")
}
func recursivenotreason(x, y, index int32, stridemax, dilmax int32, reasonable []dimlayer) (output int32, dimslayer []dimlayer, err error) {
	//	fmt.Printf("%d ,", index)
	dimslayer = copydimslayer(reasonable)
	w := dimslayer[index].w
	maxp := maxpad(w, -1)
	maxd := maxdilation(x, w, maxp, stridemax)
	maxs := maxstride(x, w, maxp, maxd, dilmax)

	for k := int32(1); k <= maxs; k++ {
		for i := int32(0); i <= maxp; i++ {
			for j := int32(1); j <= maxd; j++ {
				dimslayer[index].set(i, j, k)
				output = dimslayer[index].out(x)

				/*	if output == y && index == int32(len(reasonable)-1) {
						return output, dimslayer, nil
					}
				*/
				for l := index + 1; l < int32(len(reasonable)); l++ {
					if output < 0 {
						//		fmt.Println("Break")
						break
					}
					output = dimslayer[l].out(output)

				}
				if output == y {
					return output, dimslayer, nil
				} else if output != y && output != -1 {
					if index != int32(len(reasonable)-1) {
						op, dl, err := recursivenotreason(output, y, index+1, stridemax, dilmax, reasonable)
						if err == nil {
							output, dimslayer = op, dl
							return output, dimslayer, err
						}
					}

				}
			}
		}
	}

	return output, dimslayer, errors.New("Couldn't find anything")

}
func distancecheck(goal, actual int32) int32 {
	if actual < 0 {
		actual = -actual
	}
	return goal - actual
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
