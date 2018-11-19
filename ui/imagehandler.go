package ui

import (
	"image"
	"image/jpeg"
	"net/http"
	"sync"
)

//ImageHandler takes
type ImageHandler struct {
	img       []image.Image
	mux       sync.Mutex
	iin, iout int
	size      int
}

//MakeImageHandler makes a new image handler
func MakeImageHandler(img <-chan image.Image, buffersize chan<- int) *ImageHandler {
	size := 5
	x := &ImageHandler{}
	x.img = make([]image.Image, size)
	x.size = size
	go x.runchannel(img, buffersize)
	return x
}

func (l *ImageHandler) runchannel(img <-chan image.Image, buffersize chan<- int) {

	for im := range img {
		if (l.iin+1)%l.size != l.iout%l.size {
			l.img[l.iin%l.size] = im
			l.iin++
		}
		buffersize <- l.iin - l.iout
	}

}

//F is the function that returns the handle function
func (l *ImageHandler) Handle() func(w http.ResponseWriter, req *http.Request) {
	return func(w http.ResponseWriter, req *http.Request) {
		if (l.iout+1)%l.size != l.iin%l.size && l.img[(l.iout+1)%l.size] != nil {
			l.iout++
			check(jpeg.Encode(w, l.img[l.iout%l.size], nil))

		} else if l.img[l.iout%l.size] != nil {
			check(jpeg.Encode(w, l.img[l.iout%l.size], nil))
		}

		/*
			key := "loss"
			e := `"` + key + `"`
			w.Header().Set("Etag", e)
			w.Header().Set("Cache-Control", "no-store")
			if match := req.Header.Get("If-None-Match"); match != "" {
				if strings.Contains(match, e) {
					w.WriteHeader(http.StatusNotModified)
					return
				}
			}*/

	}

}
