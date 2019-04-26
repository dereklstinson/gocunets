package nvutil

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/npp"
)

func findPlanarChansForUint8(x *npp.Uint8, size, n int) ([]*npp.Uint8, error) {
	if size%n != 0 {
		return nil, errors.New(" size%n != 0")
	}
	xplanar := make([]*npp.Uint8, n)
	for i := 0; i < n; i++ {
		xplanar[i] = x.Offset((int32)((i * size) / n))

	}
	return xplanar, nil

}

func convertNppitoNppsCHW(channel []*npp.Uint8, sizes []npp.Size, mem *npp.Uint8) (n uint, err error) {

	var destoffset, srcsize uint

	for i := range sizes {
		h, w := sizes[i].Get()
		srcsize = uint(h * w)
		destoffset = srcsize * uint(i)

		err = nvidia.Memcpy(mem.Offset((int32)(destoffset)), channel[i], srcsize)
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
			err = nvidia.Memcpy(mem.Offset(int32((i*boffsets[i])+(j*coffsets[i][j]))), channels[i][j], (uint)(coffsets[i][j]))
			if err != nil {
				return err
			}
		}

	}
	return nil
}

func nppuint8tonppfloat32(h *Handle, src *npp.Uint8, dst *npp.Float32, length int32) error {
	return npp.Convert8u32f(src, dst, length, h.ctx)
}

func chanoffset(size npp.Size) int {
	w, h := size.Get()
	return int(w * h)
}

//Mirror - flips images according to axis. If dest is nil then function is done in place
func mirror(h *Handle, src, dest []*npp.Uint8, sizes npp.Size, flip npp.Axis) error {
	var err error
	if dest == nil {
		for i := range src {
			err = npp.Mirror8uC1IR(src[i], 0, sizes, flip, h.ctx)
			if err != nil {
				return err
			}
		}

	}
	for i := range src {
		err = npp.Mirror8uC1R(src[i], 0, dest[i], 0, sizes, flip, h.ctx)
		if err != nil {
			return err
		}
	}

	return nil
}
