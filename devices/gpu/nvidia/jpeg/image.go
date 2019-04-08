package jpeg

import (
	"io"
	"io/ioutil"

	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/nvjpeg"
)

//Image contains the data used for nvjpeg images
type Image struct {
	img      *nvjpeg.Image
	frmt     nvjpeg.OutputFormat
	channels []Channel
	pitch,height int32
}

//Channel contains a pointer to cuda memory along with Pitch and Height
type Channel struct {
	Ptr           gocu.Mem
	Pitch, Height int32
}

//GetChannels returns the channels being held by img
func (img *Image) GetChannels() []Channel {
	return img.channels
}

//CreateDestImage returns an empty Image used to place data from stream into it
func CreateDestImage(h *nvjpeg.Handle, frmt nvjpeg.OutputFormat, r io.Reader, allocator gocu.Allocator) (*Image, error) {
	data, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return createEmptyImage(h, frmt, data, allocator)
}
func createEmptyImage(handle *nvjpeg.Handle, frmt nvjpeg.OutputFormat, data []byte, allocator gocu.Allocator) (*Image, error) {
	cpnts, _, w, h, err := nvjpeg.GetImageInfo(handle, data)
	if err != nil {
		return nil, err
	}
	img, err := nvjpeg.CreateImageDest(frmt, cpnts, w, h, allocator)
	if err != nil {
		return nil, err
	}
	ptrs, pitch := img.Get()
	chans := make([]Channel, len(ptrs))
	for i := range ptrs {
		chans[i].Ptr = ptrs[i]
		chans[i].Height = h[i]
		chans[i].Pitch = (int32)(pitch[i])
	}
	return &Image{
		img:      img,
		frmt:     frmt,
		channels: chans,
	}, nil
}
func getnvjpegimagearrays(imgs []*Image) []*nvjpeg.Image {
	nvs := make([]*nvjpeg.Image, len(imgs))
	for i := range imgs {
		nvs[i] = imgs[i].img
	}
	return nvs
}
