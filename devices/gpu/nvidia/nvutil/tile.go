package nvutil

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/jpeg"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/npp"
)

//TileHelper will help tile an image
type TileHelper struct {
	srcROIs   []npp.Rect
	dstROI    npp.Rect
	srcchan   int32
	img       *jpeg.Image
	imgsize   npp.Size
	tiles     [][]*npp.Uint8
	dims      []int32
	nelements int32
}

//CreateTileHelpers creates some tilehelpers
func CreateTileHelpers(num int) []*TileHelper {
	x := make([]*TileHelper, num)
	for i := range x {
		x[i] = new(TileHelper)
	}
	return x
}

//Set sets the tilehelper
func (t *TileHelper) Set(img *jpeg.Image, tilesize npp.Size) (err error) {
	w, h := img.Size()

	t.imgsize.Set(w, h)
	t.img = img
	chans := t.img.GetChannels()
	t.srcchan = int32(len(chans))
	srcROIs, dstROI := findSrcTilesfromDstROI(t.imgsize, tilesize)
	t.srcROIs = srcROIs
	t.dstROI = dstROI
	tw, th := tilesize.Get()
	t.dims = []int32{1, t.srcchan, (int32)(len(srcROIs)), tw, th}
	t.nelements = findTileDestSize(t.srcROIs, t.dstROI, t.srcchan)
	return err
}

//BatchTileSet sets a batch of helpers with a batch of imgs
func BatchTileSet(hlprs []*TileHelper, imgs []*jpeg.Image, tilesize npp.Size) (err error) {
	if len(hlprs) != len(imgs) {
		return errors.New("TileBatchSet: len(hlprs)!=len(imgs)")
	}
	for i := range imgs {
		err = hlprs[i].Set(imgs[i], tilesize)
		if err != nil {
			return err
		}
	}
	return nil
}

//GetDestNumOfElements returns the number of elements the dest location will need
func (t *TileHelper) GetDestNumOfElements() (n int32) {
	return t.nelements
}
func findTileDestSize(srcROI []npp.Rect, dstROI npp.Rect, imagechans int32) (elements int32) {
	var areas int32
	for i := range srcROI {
		_, _, w, h := srcROI[i].Get()
		areas += w * h
	}

	return areas * imagechans
}

//BatchTileTotalElements returns the number of elements that make up the tiled images
func BatchTileTotalElements(helpers []*TileHelper) int32 {
	var adder int32
	for _, t := range helpers {
		adder += t.GetDestNumOfElements()
	}
	return adder
}

//FindDims will find dims for a tensor.  Will return nil if it can't be made. IE the dims of each helper isn't the same
func FindDims(helpers []*TileHelper) []int32 {
	batch := (int32)(len(helpers))
	if batch == 0 {
		return nil
	}
	pdims := make([]int32, 0)
	for _, h := range helpers {
		if len(pdims) == 0 {
			pdims = h.dims
		} else {
			if !compairdims(pdims, h.dims) {
				return nil
			}

		}

	}
	pdims[0] = batch
	return pdims
}
func BatchTiles(h *Handle, hlprs []*TileHelper, dest *npp.Uint8, s gocu.Streamer) error {
	var offset int32
	var err error
	for i := range hlprs {
		err = hlprs[i].TiledCTHW(h, dest.Offset(offset), s)
		if err != nil {
			return err
		}
		offset += hlprs[i].nelements

	}
	return nil
}
func compairdims(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

//TiledCTHW converts an image to an CTHW and places it into dest
func (t *TileHelper) TiledCTHW(h *Handle, dest *npp.Uint8, s gocu.Streamer) error {
	chans := t.img.GetChannels()

	_, _, wd, ht := t.dstROI.Get()
	var destsize npp.Size
	destsize.Set(wd, ht)
	destsections := make([][]*npp.Uint8, t.srcchan)
	destchanptrs, err := findPlanarChansForUint8(dest, (int)(t.nelements), (int)(t.srcchan))
	if err != nil {
		fmt.Println("First Section")
		return err
	}
	for i := range destsections {
		//	fmt.Println("Starting Next destsections")
		destsections[i], err = findOffsetsforROISinPlane(destchanptrs[i], t.srcROIs)
		if err != nil {

			fmt.Println("Second Section:")
			return err
		}
		err = s.Sync()
		if err != nil {
			return err
		}
		for j, section := range destsections[i] {

			srcchans := []*npp.Uint8{npp.MakeUint8Unsafe(chans[i].Mem().Ptr())}
			destchans := []*npp.Uint8{section}

			err = resizenpp(h, srcchans, destchans, t.imgsize, destsize, t.srcROIs[j], t.dstROI)
			if err != nil {
				return err
			}
			//err = s.Sync()
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
