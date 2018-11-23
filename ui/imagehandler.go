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
func MakeImageHandler(bufferlen int, img <-chan image.Image, buffersize chan<- int) *ImageHandler {

	x := &ImageHandler{}
	x.img = make([]image.Image, bufferlen)
	x.size = bufferlen
	go x.runchannel(img, buffersize)
	return x
}

func (l *ImageHandler) runchannel(img <-chan image.Image, buffersize chan<- int) {

	for im := range img {

		if (l.iin+1)%l.size != l.iout%l.size {
			l.mux.Lock()
			l.img[l.iin%l.size] = im
			l.mux.Unlock()
			l.iin++
		}
		buffersize <- l.iin - l.iout
	}

}

//Handle is the function that returns the handle function
func (l *ImageHandler) Handle() func(w http.ResponseWriter, req *http.Request) {
	return func(w http.ResponseWriter, req *http.Request) {
		if l.img == nil {

		} else if (l.iout+1)%l.size != l.iin%l.size && l.img[(l.iout+1)%l.size] != nil {
			l.iout++
			l.mux.Lock()
			check(jpeg.Encode(w, l.img[l.iout%l.size], nil))
			l.mux.Unlock()

		} else if l.img[l.iout%l.size] != nil {
			l.mux.Lock()
			check(jpeg.Encode(w, l.img[l.iout%l.size], nil))
			l.mux.Unlock()
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
