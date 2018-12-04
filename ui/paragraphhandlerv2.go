package ui

import (
	"fmt"
	"net/http"
	"sync"
)

//ParagraphHandlerV2 is the handler for the ui
type ParagraphHandlerV2 struct {
	par       []string
	mux       sync.Mutex
	iin, iout int
	size      int
	p         chan string
	b         chan int
}

//MakeParagraphHandlerV2 makes a new image handler
func MakeParagraphHandlerV2(bufferlen int) *ParagraphHandlerV2 {

	x := &ParagraphHandlerV2{}
	x.par = make([]string, bufferlen)
	x.size = bufferlen
	x.p = make(chan string, bufferlen)
	x.b = make(chan int, bufferlen)
	go x.runchannel(x.p, x.b)
	return x
}

//Paragraph works like Sprintf sending messages through a hidden channel
func (l *ParagraphHandlerV2) Paragraph(format string, a ...interface{}) {
	l.p <- fmt.Sprintf(format, a...)
}

//Buffer returns the buffer channel
func (l *ParagraphHandlerV2) Buffer() <-chan int {
	return l.b
}
func (l *ParagraphHandlerV2) runchannel(p <-chan string, b chan<- int) {

	for pg := range p {

		if (l.iin+1)%l.size != l.iout%l.size {
			l.mux.Lock()
			l.par[l.iin%l.size] = pg
			l.mux.Unlock()
			l.iin++
		}
		b <- l.iin - l.iout
	}

}

//Handle is the function that returns the handle function
func (l *ParagraphHandlerV2) Handle() func(w http.ResponseWriter, req *http.Request) {
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
