package nvutil

import (
	"fmt"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/jpeg"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/npp"
)

//TileHelper will help tile an image
type TileHelper struct {
	srcROIs []npp.Rect
	dstROI  npp.Rect
	srcchan int32
	img     *jpeg.Image
	imgsize npp.Size
	tiles   [][]*npp.Uint8
}

/*
func SetTileHelpers(helpers []*TileHelper, imgs []*jpeg.Image, tsizes []npp.Size, stridew, strideh []int32) (err error) {
	if len(helpers) != len(imgs) || len(helpers) != len(tsizes) || len(helpers) != len(stridew) || len(helpers) != len(strideh) {
		return errors.New("all the arrays need to be the same size")
	}
	for i, th := range helpers {
		err = th.Set(imgs[i], tsizes[i], stridew[i], strideh[i])
		if err != nil {
			return err
		}
	}
	return nil
}
*/
//Set sets the tilehelper
func (t *TileHelper) Set(img *jpeg.Image, tilesize npp.Size, stridew, strideh int32) (err error) {
	w, h := img.Size()
	var srcsize npp.Size
	srcsize.Set(w, h)
	t.imgsize.Set(w, h)
	t.img = img
	chans := t.img.GetChannels()
	t.srcchan = int32(len(chans))
	t.srcROIs, t.dstROI, err = FindSrcROIandDstROI(srcsize, tilesize, stridew, strideh)
	return err
}

//GetDestNumOfElements returns the number of elements the dest location will need
func (t *TileHelper) GetDestNumOfElements() (n int32) {
	return findTileDestSize(t.srcROIs, t.dstROI, t.srcchan)
}
func findTileDestSize(srcROI []npp.Rect, dstROI npp.Rect, imagechans int32) (elements int32) {
	//tiles := (int32)(len(srcROI))
	var areas int32
	for i := range srcROI {
		_, _, w, h := srcROI[i].Get()
		areas += w * h
	}

	return areas * imagechans
}

func (t *TileHelper) determindestinationsize() {

}

func (t *TileHelper) TiledCSHW(h *Handle, dest *npp.Uint8, destnelements int, s gocu.Streamer) error {
	chans := t.img.GetChannels()

	_, _, wd, ht := t.dstROI.Get()
	var destsize npp.Size
	destsize.Set(wd, ht)
	nchans := len(chans)
	destsections := make([][]*npp.Uint8, nchans)
	destchanptrs, err := findPlanarChansForUint8(dest, destnelements, nchans)
	if err != nil {
		return err
	}
	for i := range destsections {
		fmt.Println("Starting Next destsections")
		destsections[i], err = findPlanarChansForUint8(destchanptrs[i], destnelements/nchans, len(t.srcROIs))
		if err != nil {
			return err
		}
		err = s.Sync()
		if err != nil {
			return err
		}
		for j := range destsections[i] {

			srcchans := []*npp.Uint8{npp.MakeUint8Unsafe(chans[i].Mem().Ptr())}
			destchans := []*npp.Uint8{destsections[i][j]}

			err = resizenpp(h, srcchans, destchans, t.imgsize, destsize, t.srcROIs[j], t.dstROI)
			if err != nil {
				return err
			}
			err = s.Sync()
			if err != nil {
				fmt.Println("SourceROI is: ", t.srcROIs[j])
				fmt.Printf("error at : i: %d/%d, and j: %d/%d\n\n", i, len(destsections), j, len(destsections[i]))
				return err
			}
		}

	}
	return nil
}

//func (t TileHelper)
