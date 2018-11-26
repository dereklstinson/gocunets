package ui

import (
	"fmt"
	"net/http"
	"sync"
)

//ParagraphHandler takes
type ParagraphHandler struct {
	par       []string
	mux       sync.Mutex
	iin, iout int
	size      int
}

//MakeParagraphHandler makes a new image handler
func MakeParagraphHandler(bufferlen int, paragraph <-chan string, buffersize chan<- int) *ParagraphHandler {

	x := &ParagraphHandler{}
	x.par = make([]string, bufferlen)
	x.size = bufferlen
	go x.runchannel(paragraph, buffersize)
	return x
}

func (l *ParagraphHandler) runchannel(paragraph <-chan string, buffersize chan<- int) {

	for pg := range paragraph {

		if (l.iin+1)%l.size != l.iout%l.size {
			l.mux.Lock()
			l.par[l.iin%l.size] = "<p>" + pg + "</p>"
			l.mux.Unlock()
			l.iin++
		}
		buffersize <- l.iin - l.iout
	}

}

//Handle is the function that returns the handle function
func (l *ParagraphHandler) Handle() func(w http.ResponseWriter, req *http.Request) {
	return func(w http.ResponseWriter, req *http.Request) {
		if l.par == nil {

		} else if (l.iout+1)%l.size != l.iin%l.size {
			l.iout++
			l.mux.Lock()
			fmt.Fprintf(w, l.par[l.iout%l.size])

			l.mux.Unlock()

		} else {
			l.mux.Lock()
			fmt.Fprintf(w, l.par[l.iout%l.size])
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
