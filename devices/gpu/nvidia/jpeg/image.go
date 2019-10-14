package jpeg

import (
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/nvjpeg"
	"github.com/dereklstinson/cutil"
	"io"
	"io/ioutil"
)

//Image contains the data used for nvjpeg images
type Image struct {
	img           *nvjpeg.Image
	frmt          nvjpeg.OutputFormat
	subsampling   nvjpeg.ChromaSubsampling
	channels      []Channel
	width, height int32
}

//Size returns the pitch and height
func (img *Image) Size() (w, h int32) {
	return img.width, img.height
}

//Channel contains a pointer to cuda memory along with Pitch and Height
type Channel struct {
	ptr  cutil.Mem
	h, w int32
}

//Set sets the channels
func (c *Channel) Set(ptr cutil.Mem, w, h int32) {
	c.ptr = ptr
	c.h = h
	c.w = w
}

//Size returns the w, h values of channel
func (c *Channel) Size() (w, h int32) {
	return c.w, c.h
}

//Mem returns the gocu.Mem of the channel
func (c *Channel) Mem() cutil.Mem {
	return c.ptr
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
	chromasubsampling, w, h, err := nvjpeg.GetImageInfo(handle, data)
	if err != nil {
		return nil, err
	}
	img, err := nvjpeg.CreateImageDest(frmt, w, h, allocator)
	if err != nil {
		return nil, err
	}
	ptrs, _ := img.Get()
	chans := make([]Channel, len(ptrs))
	wactual, hactual := nvjpeg.ChannelDimHelper(frmt, w, h)
	for i := range ptrs {
		chans[i].ptr = ptrs[i]
		chans[i].h = hactual[i]

		chans[i].w = wactual[i]
	}
	return &Image{
		img:         img,
		frmt:        frmt,
		channels:    chans,
		subsampling: chromasubsampling,
		height:      h[0],
		width:       w[0],
	}, nil
}
func getnvjpegimagearrays(imgs []*Image) []*nvjpeg.Image {
	nvs := make([]*nvjpeg.Image, len(imgs))
	for i := range imgs {
		nvs[i] = imgs[i].img
	}
	return nvs
}
