package gpuperformance

import (
	"io"
	"net/http"
	"sync"

	"github.com/dereklstinson/gocunets/ui/plot"
)

//CoreClock handles the device core clock speed info
type CoreClock struct {
	plots io.WriterTo
	title string
	xaxis string
	yaxis string
	h, w  int
	data  []plot.LabeledData
	mux   sync.Mutex
}

func makeCoreClock(values <-chan []int, numberofplots, plotlengths int) *CoreClock {

	x := &CoreClock{
		title: "Device Core Speeds",
		xaxis: "Time",
		yaxis: "MHz",
		h:     6,
		w:     15,

		data: makeinitializedlabeldata(numberofplots, plotlengths),
	}

	go x.runchannel(values)

	return x
}

func (m *CoreClock) runchannel(value <-chan []int) {
	var err error
	for val := range value {
		m.mux.Lock()
		for x := range val {
			placeandshiftback(m.data[x], val[x])
		}
		m.plots, err = plot.Verses2(m.title, m.xaxis, m.yaxis, m.h, m.w, m.data)
		if err != nil {
			panic(err)
		}
		m.mux.Unlock()
	}
}

//Handle is the function that returns the handle function
func (m *CoreClock) Handle() func(w http.ResponseWriter, req *http.Request) {
	return func(w http.ResponseWriter, req *http.Request) {
		m.mux.Lock()
		if m.plots != nil {
			_, err := m.plots.WriteTo(w)
			if err != nil {
				panic(err)
			}
		}
		m.mux.Unlock()

	}

}
