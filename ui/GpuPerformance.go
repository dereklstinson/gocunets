package ui

import (
	"io"
	"sync"
	"time"

	"github.com/dereklstinson/GoCuNets/ui/plot"
	"github.com/dereklstinson/GoCuNets/utils/hwperf"
)

type Memory struct {
	device       hwperf.Device
	Plots        io.WriterTo
	ceiling      float32
	title        string
	xaxis        string
	yaxis        string
	h, w         int
	originaldata [][]float32
	data         []plot.LabeledData
	index        int
	epoc         bool
	mux          sync.Mutex
}
type CoreClocks struct {
	device       hwperf.Device
	Plots        io.WriterTo
	ceiling      float32
	title        string
	xaxis        string
	yaxis        string
	h, w         int
	originaldata [][]float32
	data         []plot.LabeledData
	index        int
	epoc         bool
	mux          sync.Mutex
}
type MemClocks struct {
	device       hwperf.Device
	Plots        io.WriterTo
	ceiling      float32
	title        string
	xaxis        string
	yaxis        string
	h, w         int
	originaldata [][]float32
	data         []plot.LabeledData
	index        int
	epoc         bool
	mux          sync.Mutex
}
type Temp struct {
	device       hwperf.Device
	Plots        io.WriterTo
	ceiling      float32
	title        string
	xaxis        string
	yaxis        string
	h, w         int
	originaldata [][]float32
	data         []plot.LabeledData
	index        int
	epoc         bool
	mux          sync.Mutex
}
type Power struct {
	device       hwperf.Device
	Plots        io.WriterTo
	ceiling      float32
	title        string
	xaxis        string
	yaxis        string
	h, w         int
	originaldata [][]float32
	data         []plot.LabeledData
	index        int
	epoc         bool
	mux          sync.Mutex
}

func CreateGPUPerformanceHandlers(refreshms, Lengthoftime int) (*Memory, *CoreClocks, *MemClocks, *Temp, *Power) {

	return &Memory{}, &CoreClocks{}, &MemClocks{}, &Temp{}, &Power{}
}
func runchannel(device []*hwperf.Device, refreshms int, mem, clockmem, clockcore, temp, power chan<- []int) {
	ticker := time.NewTicker(time.Duration(refreshms) * time.Millisecond)
	dl := len(device)
	devcoreclocks := make([]int, dl)
	devmemclocks := make([]int, dl)
	memused := make([]int, dl)
	//memfree := make([]int, dl)
	powers := make([]int, dl)
	temps := make([]int, dl)

	for {
		<-ticker.C
		for i := range device {
			device[i].RefreshStatus()
			dc, dm := device[i].Clocks()
			devcoreclocks[i], devmemclocks[i] = int(dc), int(dm)
			temps[i] = int(device[i].Temp())
			powers[i] = int(device[i].Power())
			usedm, _ := device[i].Memory()
			memused[i] = int(usedm)

		}
		mem <- memused
		temp <- temps
		power <- powers
		clockcore <- devcoreclocks
		clockmem <- devmemclocks

	}

}

/*
//Handle is the function that returns the handle function
func (l *Memory) Handle() func(w http.ResponseWriter, req *http.Request) {
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

	}

}
*/
