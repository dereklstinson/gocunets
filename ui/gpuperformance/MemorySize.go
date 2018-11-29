package gpuperformance

import (
	"io"
	"net/http"
	"strconv"
	"sync"

	"github.com/dereklstinson/GoCuNets/ui/plot"
	"gonum.org/v1/plot/plotter"
)

type Memory struct {
	Plots io.WriterTo
	title string
	xaxis string
	yaxis string
	h, w  int
	data  []plot.LabeledData
	mux   sync.Mutex
}

func makeMemory(values <-chan []int, numberofplots, plotlengths int) *Memory {

	x := &Memory{
		title: "DeviceMem",
		xaxis: "Time",
		yaxis: "MB (used)",
		h:     6,
		w:     15,

		data: makeinitializedlabeldata(numberofplots, plotlengths),
	}

	go x.runchannel(values)

	return x
}

func (m *Memory) runchannel(value <-chan []int) {
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
func (m *Memory) Handle() func(w http.ResponseWriter, req *http.Request) {
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
func makeinitializedlabeldata(numberofplots int, lengthofplots int) []plot.LabeledData {
	data := make([]plot.LabeledData, numberofplots)
	for i := range data {
		data[i].Label = "Device" + strconv.Itoa(i)
		data[i].Data = make(plotter.XYs, lengthofplots)
		for j := range data[i].Data {
			data[i].Data[j].X = float64(j)
		}
	}
	return data
}
