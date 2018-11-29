package gpuperformance

import (
	"io"
	"net/http"
	"sync"

	"github.com/dereklstinson/GoCuNets/ui/plot"
)

type MemClock struct {
	Plots io.WriterTo
	title string
	xaxis string
	yaxis string
	h, w  int
	data  []plot.LabeledData
	mux   sync.Mutex
}

func makeMemClock(values <-chan []int, numberofplots, plotlengths int) *MemClock {

	x := &MemClock{
		title: "DeviceMemClock",
		xaxis: "Time",
		yaxis: "MHZ",
		h:     6,
		w:     15,
		data:  makeinitializedlabeldata(numberofplots, plotlengths),
	}
	go x.runchannel(values)
	return x
}

func (m *MemClock) runchannel(value <-chan []int) {
	var err error
	for val := range value {
		m.mux.Lock()
		for x := range val {
			placeandshiftback(m.data[x], val[x])
		}
		m.Plots, err = plot.Verses2(m.title, m.xaxis, m.yaxis, m.h, m.w, m.data)
		if err != nil {
			panic(err)
		}
		m.mux.Unlock()
	}
}

//Handle is the function that returns the handle function
func (m *MemClock) Handle() func(w http.ResponseWriter, req *http.Request) {
	return func(w http.ResponseWriter, req *http.Request) {
		m.mux.Lock()
		if m.Plots != nil {
			_, err := m.Plots.WriteTo(w)
			if err != nil {
				panic(err)
			}
		}
		m.mux.Unlock()

	}

}
