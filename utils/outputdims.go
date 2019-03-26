package utils

import "errors"

type convprop struct {
	input, weight, output int32

	s, p, d int32
}
type ConvolutionCombo struct {
	Output                []int32
	Stride, Pad, Dilation []int32
}

/*
func FindAllCombos(NCHW bool, tensor, weights []int32) []ConvolutionCombo {
	tenlen := len(tensor)
	convoptions := tenlen - 2
	stridemax := make([]int32, convoptions)
	stride := make([]int32, convoptions)
	pad := make([]int32, convoptions)
	padmax := make([]int32, convoptions)
	dilation := make([]int32, convoptions)
	dilationmax := make([]int32, convoptions)
	output := make([]int32, tenlen)
	output[0] = tensor[0]

	outputworkingsection := make([]int32, convoptions)
	weightworkingsection := make([]int32, convoptions)
	inputworkingsection := make([]int32, convoptions)
	if NCHW {
		output[1] = weights[0]
		for i := range inputworkingsection {
			inputworkingsection[i] = tensor[2+i]
			weightworkingsection[i] = weights[2+i]
		}
	} else {
		output[tenlen-1] = weights[0]
		for i := range inputworkingsection {
			inputworkingsection[i] = tensor[1+i]
			weightworkingsection[i] = weights[1+i]
		}
	}
	return nil
}
*/

type xwcombo struct {
	x, w               int32
	padstridedilations [][]int32
}

func (xc *xwcombo) check(x, w int32) bool {
	if xc.w == w && xc.x == x {
		return true
	}
	return false
}

/*
func FindMinOutputs(x, w []int32, NCHW bool) (ccs []ConvolutionCombo, err error) {
	wx, ww, wo := findworkingvalues(x, w, NCHW)

	xwcombs:=make([]xwcombo,0)
	combos:=make([]ConvolutionCombo,0)

	for i := range wx {
		var flag bool
		td,wd:=wx[i],ww[i]
		for i:=range xwcombs{
			if xwcombs[i].check(td,wd){
				flag=true
			}
		}
		if !flag{
			var xwcomb xwcombo
			xwcomb.x,xwcomb.w=td,wd
			minval:=int32(9999999999)
		}
		mpad:=maxpad(xw[i])
		for h:=0;h<mpad;h++{


		maxdil:=maxdilation(td,wd,h)

		for j:=maxdil;j>=1;j--{
			mstride:=maxstride(td,wd,h,j)

				k:=mstride;k>=1;k--{
					val:=findoutputdim(td,wd,k,h,j)
				if val!=-1 &&val<=minval{
xwcombo.padstridedilations=append(xwcombo.padstridedilations,[]int32{h,k,j})
				}

				}


		}
	}
	xwcombs=append(xwcombs,xwcomb)
}
dimhascombo:=make([]int32,len(wx))
for i:=range wx{
	td,wd:=wx[i],ww[i]
	for j:=range xwcombs{
		if xwcombs[j].check(td,wd){
			dimhascombo[i]=j
		}
	}
}
for i:=range xwcombs{}
}
*/

func (c *ConvolutionCombo) OutputVol() int32 {
	return FindVolumeInt32(c.Output, nil)
}
func FindMaxOutput(x, w []int32, NCHW bool) (cc ConvolutionCombo, err error) {
	wx, ww, wo := findworkingvalues(x, w, NCHW)
	dilation := make([]int32, len(wx))
	padding := make([]int32, len(wx))
	stride := make([]int32, len(wx))
	for i := range wx {
		mp := maxpad(ww[i])
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

func maxpad(w int32) int32 {
	return w - 1
}

//maxstride can't be larger than the weights itself(self imposed)
func maxstride(x, w, p, d int32) int32 {
	if (((w - 1) * d) + 1) > x+(2*p) {
		return -1
	}
	if (((w - 1) * d) + 1) == x+(2*p) {
		return 1
	}
	val := x + (2 * p) - (((w - 1) * d) + 1)
	if val < w {
		return val
	}
	return w
}

//max dilation can't be larger than weights(self imposed)
func maxdilation(x, w, p int32) int32 {
	val := (x + (2 * p) - 1) / (w - 1)
	if val < w {
		return val
	}
	return w
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
