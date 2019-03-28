package utils

type convprop struct {
	input, weight, output int32

	s, p, d int32
}
type xwcombo struct {
	x, w      int32
	pds       [][]int32
	minoutval int32
}

func findworkingpathvalues(x, y []int32, w [][]int32, NCHW bool) (wx, wy []int32, ww [][]int32) {
	convoptions := len(x) - 2
	wx = make([]int32, convoptions)
	ww = make([][]int32, len(w))
	wy = make([]int32, convoptions)
	for i := range ww {
		ww[i] = make([]int32, convoptions)
	}
	if NCHW {
		for i := range wx {
			wx[i] = x[2+i]
			wy[i] = y[2+i]
			for j := range ww {
				ww[j][i] = w[j][2+i]
			}

		}
	} else {
		for i := range wx {
			wx[i] = x[1+i]
			wy[i] = y[1+i]

			for j := range ww {
				ww[j][i] = w[j][1+i]
			}
		}
	}
	return wx, wy, ww
}

func (xc *xwcombo) check(x, w int32) bool {
	if xc.w == w && xc.x == x {
		return true
	}
	return false
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
func findreverseoutputdim(x, w, s, p, d int32) int32 {
	// output = 1+ ((input + (2*padding) - (((filter-1)*dilation)+1))/slide)
	// *now input<==>output
	//input=  1+ ((output + (2*padding) - (((filter-1)*dilation)+1))/slide)
	//	(input-1)*slide = (output +2*padding)-(((filter-1)*dilation)+1)
	//output= 2*padding-(((filter-1)*dilation)+1)-(input-1)*slide
	// input = 1 + (output + (2*padding) - (((filter-1)*dilation)+1))/slide
	//  slide *(input-1) = output + (2*padding) - (((filter-1)*dilation)+1)
	//  output = (slide *(input-1)) - (2*padding) + (((filter-1)*dilation)+1)
	return (s * (x - 1)) - (2 * p) + (((w - 1) * d) + 1)
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
