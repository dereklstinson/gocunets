//Package nvutil are functions that use the other nvidia packages and allows them to be used with each other
package nvutil

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/jpeg"

	"github.com/dereklstinson/GoCudnn/npp"
)

//ImageToNppi Converts an jpeg.Image to a npp.type with its npp.Size
func ImageToNppi(img *jpeg.Image) (planar []*npp.Uint8, size npp.Size) {
	chans := img.GetChannels()
	planar = make([]*npp.Uint8, len(chans))
	w, h := img.Size()
	size.Set(w, h)
	for i := range chans {
		planar[i] = (*npp.Uint8)(chans[i].Mem().Ptr())

	}
	return planar, size

}

type Handle struct {
	ctx      *npp.StreamContext
	polation npp.InterpolationMode
}

func CreateHandle(ctx *npp.StreamContext, polation npp.InterpolationMode) *Handle {
	return &Handle{
		ctx:      ctx,
		polation: polation,
	}
}
func resizenpp(h *Handle, src, dest []*npp.Uint8, srcSize, destSize npp.Size, srcROI, destROI npp.Rect) (err error) {
	ws, _ := srcSize.Get()
	wd, _ := destSize.Get()
	switch len(src) {
	case 1:
		err = npp.Resize8uC1R(src[0], srcSize, ws, srcROI, dest[0], destSize, wd, destROI, h.polation, h.ctx)
		if err != nil {
			fmt.Println("Error in case 1: Resize8uC1R")
		}
		return err
	case 3:
		err = npp.Resize8uP3R(src, srcSize, ws, srcROI, dest, destSize, wd, destROI, h.polation, h.ctx)
		if err != nil {
			fmt.Println("Error in case 3: Resize8uP3R")
		}
		return err
	case 4:
		err = npp.Resize8uP4R(src, srcSize, ws, srcROI, dest, destSize, wd, destROI, h.polation, h.ctx)
		if err != nil {
			fmt.Println("Error in case 4: Resize8uP4R")
		}
		return err
	}
	return errors.New("Unsupported src,dest size")
}

func cudnnbatchchannelsize(x *tensor.Volume) (batch int, channel int, err error) {
	frmt, _, dims, err := x.Properties()
	if err != nil {
		return 0, 0, err
	}

	fflg := frmt
	switch frmt {
	case fflg.NCHW():
		return int(dims[0]), int(dims[1]), nil
	case fflg.NHWC():
		return int(dims[0]), int(dims[len(dims)-1]), nil

	default:
		return -1, -1, errors.New("Unsupported Format")

	}
}

func finddimpadandsections(src, dst, stride int32) (padL, padU int32) {
	nstride := ((src - dst) / stride) //number of strides that start within the the src.
	fmt.Println("nstride: ", nstride)
	fmt.Println("nStride Remainder: ", (src-dst)%stride)
	laststridestart := (nstride) * stride
	fmt.Println("Laststride: ", laststridestart)
	padding := src - (laststridestart + dst)
	//	sections = intceiling(src+padding, stride)
	if padding == 0 {
		return 0, 0
	}
	if padding < 0 {
		panic("padding <0")
	}
	padL = padding / 2
	padU = padL
	if padding%2 == 1 {
		if rand.Int31n(1) == 1 {
			padU++
		} else {
			padL++
		}
	}
	//fmt.Println(padL, padU)
	return -padL, padU
}

func intceiling(a, b int32) int32 {
	return ((a - int32(1)) / b) + int32(1)
}

//FindSrcROIandDstROI will return an array of srcROIs that will be used to tile the dstsize.
func FindSrcROIandDstROI(src, dst npp.Size, strideW, strideH int32) (srcROI []npp.Rect, dstROI npp.Rect, err error) {
	srcW, srcH := src.Get()
	dstW, dstH := dst.Get()
	padT, padB := finddimpadandsections(srcH, dstH, strideH)
	padL, padR := finddimpadandsections(srcW, dstW, strideW)
	fmt.Println("Padding --> padT,padB,padL,padR", padT, padB, padL, padR)
	srcROI = make([]npp.Rect, 0)
	dstROI.Set(0, 0, dstW, dstH)
	//for i := padT; i < srcH+padB-(dstH+strideH); i += strideH {
	//	for j := padL; j < srcW+padR-(dstW+strideW); j += strideW {

	//	bottompad := (srcH - dstH) - padB
	//	rightpad := (srcW - dstW) - padR
	for i := padT; i < srcH+padB-(dstH); i += strideH {
		for j := padL; j < srcW+padR-(dstW); j += strideW {
			var (
				tempW int32
				tempH int32
				hflag bool
				wflag bool
			)

			if i < 0 {
				tempH = dstH + i
				i = 0

			} else if i+dstH+strideH > srcH {
				tempH = srcH - i
				//fmt.Println("Got to H")
				hflag = true
			} else {
				tempH = dstH
			}
			if j < 0 {
				tempW = dstW + j
				j = 0
			} else if j+dstW+strideW > srcW {
				tempW = srcW - j
				//fmt.Println("Got To W")
				wflag = true
			} else {
				tempW = dstW
			}
			if hflag && wflag {
				fmt.Println("The Corner", tempH, tempW, i, j)
			}

			var srcrect npp.Rect
			srcrect.Set(j, i, tempW, tempH)
			srcROI = append(srcROI, srcrect)

		}
	}

	return srcROI, dstROI, nil
}

func totalVol(sizes [][]npp.Size) int {

	var h, w, adder int32
	for i := range sizes {
		for j := range sizes[i] {
			w, h = (sizes[i][j].Get())
			adder += (int32)(w * h)
		}
	}

	return (int)(adder)
}
