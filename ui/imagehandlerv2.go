package ui

import (
	"image"
	"image/jpeg"
	"net/http"
	"sync"
)

//ImageHandlerV2 takes
type ImageHandlerV2 struct {
	img       []image.Image
	mux       sync.Mutex
	iin, iout int
	size      int
	imgc      chan image.Image
	buf       chan int
}

//MakeImageHandlerV2 makes a new image handler
func MakeImageHandlerV2(bufferlen int) *ImageHandlerV2 {

	x := &ImageHandlerV2{}
	x.img = make([]image.Image, bufferlen)
	x.size = bufferlen
	x.imgc = make(chan image.Image, bufferlen)
	x.buf = make(chan int, bufferlen)
	go x.runchannel(x.imgc, x.buf)
	return x
}

//Image works like Sprintf sending messages through a hidden channel
func (l *ImageHandlerV2) Image(image image.Image) {
	l.imgc <- image

}

//Buffer returns the buffer channel
func (l *ImageHandlerV2) Buffer() <-chan int {
	return l.buf
}
func (l *ImageHandlerV2) runchannel(img <-chan image.Image, buffersize chan<- int) {

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
func (l *ImageHandlerV2) Handle() func(w http.ResponseWriter, req *http.Request) {
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
