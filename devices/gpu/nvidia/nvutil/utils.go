//Package nvutil are functions that use the other nvidia packages and allows them to be used with each other
package nvutil

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/jpeg"
	gocudnn "github.com/dereklstinson/GoCudnn"

	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/npp"
)

//ImageToNppi Converts an jpeg.Image to a npp.type with its npp.Size
func ImageToNppi(img *jpeg.Image) (planar []*npp.Uint8, sizes []npp.Size) {
	chans := img.GetChannels()
	planar = make([]*npp.Uint8, len(chans))
	sizes = make([]npp.Size, len(chans))
	for i := range chans {
		planar[i] = (*npp.Uint8)(chans[i].Ptr.Ptr())
		sizes[i].Set(chans[i].Height, chans[i].Pitch)
	}
	return planar, sizes

}

type BatchMaker struct {
	memraw   *npp.Uint8
	t        *gocudnn.TensorD
	memfloat *npp.Float32
	length   int
}

func convertNppitoNppsCHW(channel []*npp.Uint8, sizes []npp.Size, mem *npp.Uint8) (n uint, err error) {

	var destoffset, srcsize uint

	for i := range sizes {
		h, w := sizes[i].Get()
		srcsize = uint(h * w)
		destoffset = srcsize * uint(i)

		err = nvidia.Memcpy(gocu.Offset(mem, (destoffset)), channel[i], srcsize)
		if err != nil {
			return n, err
		}
		n += srcsize
	}
	return n, nil
}
func convertsCHWstoNCHW(srcs []*npp.Uint8, srcsSIBs []uint, dest *npp.Uint8) (n uint, err error) {
	var destoffset uint
	for i := range srcs {
		destoffset = srcsSIBs[i] * uint(i)
		err = nvidia.Memcpy(gocu.Offset(dest, destoffset), srcs[i], srcsSIBs[i])
		if err != nil {
			return n, err
		}
		n += srcsSIBs[i]
	}
	return n, nil
}
func convertNppitoNppsNCHW(channels [][]*npp.Uint8, sizes [][]npp.Size, mem *npp.Uint8) error {
	coffsets := make([][]int, 0)
	boffsets := make([]int, 0)
	var err error
	for i := range sizes {
		var adder int
		coffsets[i] = make([]int, len(sizes[i]))
		for j := range sizes[i] {
			adder += chanoffset(sizes[i][j])
			coffsets[i] = append(coffsets[i], adder)
		}
		boffsets = append(boffsets, adder)
	}

	for i := range channels {

		for j := range channels[i] {
			err = nvidia.Memcpy(gocu.Offset(mem, uint((i*boffsets[i])+(j*coffsets[i][j]))), channels[i][j], (uint)(coffsets[i][j]))
			if err != nil {
				return err
			}
		}

	}
	return nil
}

func nppuint8tonppfloat32(src *npp.Uint8, dst *npp.Float32, length int32) error {
	return npp.Convert8u32f(src, dst, length)
}

func chanoffset(size npp.Size) int {
	w, h := size.Get()
	return int(w * h)
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
