package outputdim

import (
	"errors"
	"fmt"
	"sync"

	"github.com/dereklstinson/gocunets/utils"
	"github.com/dereklstinson/gocunets/utils/outputdim/dimlayer"
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
func (w *workinglayer) append(dl dimlayer.DimLayer) {
	wt, p, d, s := dl.Get()
	w.w = append(w.w, wt)
	w.p = append(w.p, p)
	w.d = append(w.d, d)
	w.s = append(w.s, s)
}
func (w *workinglayer) get() (ww, wp, wd, ws []int32) {
	return w.w, w.p, w.d, w.s
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
func FindMaxOutputPath(x []int32, layers [][]int32, NCHW bool) (output []int32, ccs []ConvolutionCombo, err error) {
	ccs = make([]ConvolutionCombo, 0)
	output = make([]int32, len(x))
	copy(output, x)
	for i := range layers {
		cc, err := FindMaxOutput(output, layers[i], NCHW)
		if err != nil {
			return nil, nil, err
		}
		ccs = append(ccs, cc)
		output = cc.Output

	}
	return output, ccs, nil
}

//FindReasonAbleCombosForPath not operational
func FindReasonAbleCombosForPath(x, y []int32, layers [][]int32, NCHW bool, minp, maxs, maxd int32, favordilation bool) (css []ConvolutionSettings, err error) {
	fmt.Println("Starting")
	wx, wy, ww := findworkingpathvalues(x, y, layers, NCHW)
	dimpaths := makedimpaths(wx, wy, ww)
	workingoutputs := make([]int32, len(wx))
	dimpathlayers := make([][]dimlayer.DimLayer, len(wx))
	var wg sync.WaitGroup
	errchans := make([]chan error, len(wx))
	for i, dim := range dimpaths {
		errchans[i] = make(chan error, 1)
		errchan := errchans[i]
		wg.Add(1)
		go func(i int, dim dimpath, errchan chan error) {

			var err error
			output, dpath, err := reasonablepath(dim.x, dim.y, minp, maxs, maxd, dim.path, favordilation)
			if err != nil {
				var err2 error
				fmt.Println("Doing Recursive Reason")

				outputs, dpaths, err2 := recursivenotreason(dim.x, dim.y, 0, maxs, maxd, dpath)
				fmt.Println(outputs)
				if err2 != nil {
					errchan <- err2
				}

				workingoutputs[i], dimpathlayers[i] = outputs, dpaths
				dimpathlayers[i] = dpaths

			} else {
				workingoutputs[i], dimpathlayers[i] = output, dpath

			}
			close(errchan)
			wg.Done()

		}(i, dim, errchan)
	}
	wg.Wait()
	for _, errchan := range errchans {
		for err := range errchan {
			if err != nil {
				return nil, err
			}
		}

	}
	wlayer := make([]workinglayer, len(layers))
	for i := range wlayer {
		wlayer[i] = createworkinglayer()

		for j := range dimpathlayers {
			wlayer[i].append(dimpathlayers[j][i])
		}
	}

	css = make([]ConvolutionSettings, len(layers))
	for j := range css {

		css[j] = createconvolutionsettings(wlayer[j].get())
	}

	return css, nil
}
func distancecheck(goal, actual int32) int32 {
	if actual < 0 {
		actual = -actual
	}
	return goal - actual
}

//reasonablepath tries to find a path where all the layers share the same pad, stride and dim. The deeper the network the more likely this is to work.
//if minp is larger than the maxp in one of the layers. Then minp will be set to zero. This is to maximize the chance that this will work correctly.
//Error will return if it couldn't find  a path  that share all the same vals, but it will return the closest.  with the least amount of padding,stride,and dilation
func reasonablepath(x, y, minp, maxs, maxd int32, w []int32, favordilation bool) (output int32, dimlayers []dimlayer.DimLayer, err error) {
	dimlayers = make([]dimlayer.DimLayer, len(w))
	closestdimlayers := make([]dimlayer.DimLayer, len(w))
	smallestmaxp := int32(9999999)

	for i := range w {
		closestdimlayers[i] = dimlayer.CreateDimLayer(w[i], 1, 1, 1)
		dimlayers[i] = dimlayer.CreateDimLayer(w[i], 1, 1, 1)
		if w[i]-1 < smallestmaxp {
			smallestmaxp = w[i] - 1
		}
	}
	if smallestmaxp < minp {
		minp = 0
	}
	smallestmaxd := int32(99999)
	for i := range w {
		dilmax := maxdilation(w[i], maxd)
		if smallestmaxd > dilmax {
			smallestmaxd = dilmax
		}
	}
	smallestmaxs := int32(99999)
	for i := range w {
		stridex := maxstride(w[i], maxs)
		if smallestmaxs > stridex {
			smallestmaxs = stridex
		}
	}

	closestdist := int32(9999999)
	closestchecker := int32(0)
	if favordilation {
		for i := smallestmaxp; i >= minp; i-- {
			for j := smallestmaxd; j >= int32(1); j-- {
				for k := smallestmaxs; k >= int32(1); k-- {
					output = x
					for l := range dimlayers {
						dimlayers[l].Set(i, j, k)
						output = dimlayers[l].Out(output)
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
							closestdimlayers[l].Set(i, j, k)
							output = dimlayers[l].Out(output)
						}

					}

				}
			}
		}
	} else {
		for i := smallestmaxp; i >= minp; i-- {
			for j := smallestmaxs; j >= int32(1); j-- {
				for k := smallestmaxd; k >= int32(1); k-- {
					output = x
					for l := range dimlayers {
						dimlayers[l].Set(i, k, j)
						output = dimlayers[l].Out(output)
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
							closestdimlayers[l].Set(i, k, j)
							output = dimlayers[l].Out(output)
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
func mostreasonableunreasonable(x, y int32, reasonable []dimlayer.DimLayer) (output int32, dimslayer []dimlayer.DimLayer, err error) {
	dimslayer = dimlayer.CopyDimsLayer(reasonable)
	closestdimlayers := make([]dimlayer.DimLayer, len(reasonable))

	closestdist := int32(9999999)
	closestchecker := int32(0)
	for h := range dimslayer {
		dimslayer = dimlayer.CopyDimsLayer(reasonable)
		w, _, _, _ := dimslayer[h].Get()
		maxp := maxpad(w, -1)
		maxd := maxdilation(w, -1)
		maxs := maxstride(x, -1)

		for i := int32(0); i <= maxp; i++ {
			for j := int32(1); j <= maxd; j++ {
				for k := int32(1); k <= maxs; k++ {
					dimslayer[h].Set(i, j, k)
					output = dimslayer[h].Out(x)
					if output < 0 {
						break
					}
					for l := range dimslayer[h+1:] {
						output = dimslayer[l].Out(output)

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
								closestdimlayers[l].Set(i, j, k)
								output = dimslayer[l].Out(output)
							}

						}
					}

				}
			}
		}
	}
	return output, closestdimlayers, errors.New("Couldn't find anything")
}

func recursivenotreason(x, y, padmin, stridemax, dilmax int32, current []dimlayer.DimLayer) (int32, []dimlayer.DimLayer, error) {
	hlper := make([]*dimlayer.Ofhlpr, len(current))
	for i := range current {
		w, _, _, _ := current[i].Get()
		mp := maxpad(w, w-1)
		minp := minpad(w, padmin)
		smax := maxstride(w, stridemax)
		dmax := maxdilation(w, dilmax)
		hlper[i] = dimlayer.Makeoutputfinderhelper(int32(i), int32(len(current)), x, y, 1, 1, minp, smax, dmax, mp, &current[i])
	}
	if int(x)%len(current) != 0 {
		return -1, nil, errors.New("len of current needs to be a common factor of x")
	}
	output, err := dimlayer.Backwardspdv2(x, y, hlper)
	if err != nil {
		return -1, nil, err
	}
	return output, current, nil
	//return output, nil, err
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
				mind := mindilation(wd, mindil)
				maxd := maxdilation(wd, maxdil)
				for k := mind; k <= maxd; k++ {
					mins := minstride(wd, minstrides)
					maxs := maxstride(wd, maxstrides)
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
				mind := mindilation(wd, mindil)
				maxd := maxdilation(wd, maxdil)
				for k := mind; k <= maxd; k++ {
					mins := minstride(wd, minstrides)
					maxs := maxstride(wd, maxstrides)
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
	return utils.FindVolumeInt32(c.Output, nil)
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

/*
func backwardspd(x, y, index0goalx, lastindexgoaly, index, padmin, stridemax, dilmax int32, current []dimlayer) error {

	w := current[index].w
	maxp := maxpad(w, -1)
	minp := minpad(w, padmin)
	maxd := maxdilation(w, stridemax)
	maxs := maxstride(w, dilmax)

	if index == int32(len(current)-1) {
		current[index].set(0, 1, 1)
		output := index0goalx
		for i := range current {
			output = current[i].out(output)
			if output < 1 {
				break
			}
		}
		if output == lastindexgoaly {
			return nil
		}

		y = lastindexgoaly

		return backwardspd(x, output, index0goalx, lastindexgoaly, index-1, padmin, stridemax, dilmax, current)
	} else if index == 0 {
		x = index0goalx
		return forwardspd(x, y, index0goalx, lastindexgoaly, index, padmin, stridemax, dilmax, current)
	}
	for k := int32(1); k <= maxs; k++ {
		for j := int32(1); j <= maxd; j++ {
			for i := minp; i <= maxp; i++ {
				output := findreverseoutputdim(y, w, k, i, j)
				if withinboundsx(index, output, index0goalx, lastindexgoaly) {
					current[index].set(i, j, k)
					return backwardspd(x, output, index0goalx, lastindexgoaly, index-1, padmin, stridemax, dilmax, current)

				}
			}
		}
	}

	return fmt.Errorf("backward reached end of the line at index %d", index)
}

func forwardspd(x, y, index0goalx, lastindexgoaly, index, padmin, stridemax, dilmax int32, current []dimlayer) error {

	w := current[index].w
	maxp := maxpad(w, -1)
	maxd := maxdilation(w, stridemax)
	maxs := maxstride(w, dilmax)

	if index == int32(len(current)-1) {
		y = lastindexgoaly
		return backwardspd(x, y, index0goalx, lastindexgoaly, index, padmin, stridemax, dilmax, current)

	} else if index == 0 {
		x = index0goalx
		output := index0goalx
		for i := range current {
			output = current[i].out(output)
			if output < 1 {
				break
			}
		}
		if output == lastindexgoaly {
			return nil
		}
		output = current[0].out(output)

	}

	for k := maxs; k >= int32(1); k-- {
		for j := maxd; j >= int32(1); j-- {
			for i := maxp; i >= padmin; i-- {
				output := findoutputdim(x, w, k, i, j)
				if withinboundsy(index, output, index0goalx, lastindexgoaly) {
					current[index].set(i, j, k)
					return forwardspd(output, y, index0goalx, lastindexgoaly, index+1, padmin, stridemax, dilmax, current)
				}
			}
		}
	}
	return fmt.Errorf("forward reached end of the line at index %d", index)

}
*/
