package nvutil

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/jpeg"
	"github.com/dereklstinson/GoCudnn/npp"
)

//TileHelper will help tile an image
type TileHelper struct {
	srcROIs []npp.Rect
	dstROI  npp.Rect
	srcchan int32
	img     *jpeg.Image
}

//Set sets the tilehelper
func (t *TileHelper) Set(img *jpeg.Image, tilesize npp.Size, stridew, strideh int32) (err error) {
	w, h := img.Size()
	var srcsize npp.Size
	srcsize.Set(w, h)
	chans := img.GetChannels()
	t.srcchan = int32(len(chans))
	t.srcROIs, t.dstROI, err = FindSrcROIandDstROI(srcsize, tilesize, stridew, strideh)
	return err
}

//GetDestNumOfElements returns the number of elements the dest location will need
func (t *TileHelper) GetDestNumOfElements() (n int32) {
	return findTileDestSize(t.srcROIs, t.dstROI, t.srcchan)
}
func findTileDestSize(srcROI []npp.Rect, dstROI npp.Rect, imagechans int32) (elements int32) {
	tiles := (int32)(len(srcROI))
	_, _, w, h := dstROI.Get()
	return w * h * tiles * imagechans
}

func (t *TileHelper) determindestinationsize() {

}

func (t *TileHelper) TiledCSHW(h *Handle, dest *npp.Uint8, destnelements int) error {
	chans := t.img.GetChannels()

	var srcsize npp.Size
	srcsize.Set(t.img.Size())
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
		roiptrs, err := findPlanarChansForUint8(destchanptrs[i], destnelements/nchans, len(t.srcROIs))
		destsections[i] = append(destsections[i], roiptrs...)
		for j := range destsections[i] {
			if err != nil {
				return err
			}
			srcchans := []*npp.Uint8{(*npp.Uint8)(chans[i].Ptr.Ptr())}
			destchans := []*npp.Uint8{destsections[i][j]}

			err = resizenpp(h, srcchans, destchans, srcsize, destsize, t.srcROIs[j], t.dstROI)
			if err != nil {
				return err
			}
		}

	}
	return nil
}
