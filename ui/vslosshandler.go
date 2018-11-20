package ui

import (
	"io"
	"net/http"
	"sync"

	"github.com/dereklstinson/GoCuNets/ui/plot"
)

//VSLossHandler draws a vs chart for loss.  You might want to put several things in it like Training Loss Vs Testing Loss.
type VSLossHandler struct {
	Plots        io.WriterTo
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

//LabelFloat is used to pass data through channels to the VSLossHandler
type LabelFloat struct {
	Label string
	Data  []float32
}

//NewVSLossHandle Makes a VSLoss Handle.  If epoc is true it will label the x axis as epoc. Otherwise it will label it batch.
// Just make sure you are passing the right amount of data through the LossData Channel.  So, the axis is labeled correctly.
// Y axis will be labeled Loss.
func NewVSLossHandle(title string, LossData <-chan []LabelFloat, epoc bool, plotlengths int, labels ...string) (*VSLossHandler, []LabelFloat) {
	var xaxis string
	if epoc == true {
		xaxis = "Epocs"
	} else {
		xaxis = "Batch"
	}
	numberofplots := len(labels)
	y := make([][]float32, numberofplots)
	data := make([]plot.LabeledData, numberofplots)
	x := &VSLossHandler{
		epoc:         epoc,
		title:        title,
		xaxis:        xaxis,
		yaxis:        "Loss",
		h:            6,
		w:            15,
		originaldata: y,
		data:         data,
	}

	go x.runchannel(LossData)
	lblflt := make([]LabelFloat, numberofplots)
	for i := range lblflt {
		lblflt[i].Data = make([]float32, plotlengths)
		lblflt[i].Label = labels[i]
	}
	return x, lblflt
}

func (l *VSLossHandler) runchannel(LossData <-chan []LabelFloat) {

	var err error
	for array := range LossData {
		go func(array []LabelFloat) {
			l.mux.Lock()
			for i := range array {
				for j := range array[i].Data {
					if array[i].Data[j] > 50 {
						array[i].Data[j] = 50
					}
				}
				l.originaldata[i] = append(l.originaldata[i], array[i].Data...)
				l.data[i], err = plot.NewLabeledData(array[i].Label, l.originaldata[i])

			}
			l.Plots, err = plot.Verses2(l.title, l.xaxis, l.yaxis, l.h, l.w, l.data)
			if err != nil {
				panic(err)
			}
			l.mux.Unlock()
		}(array)
		//l.data[0].
		//plot.NewLabeledData
	}
}

//ChangeHW changes the Height and Width of the Plot default is h= 6 cm, and w = 12 cm
func (l *VSLossHandler) ChangeHW(h, w int) {
	l.mux.Lock()
	l.h, l.w = h, w
	l.mux.Unlock()

}

//F is the func used for the web handler
func (l *VSLossHandler) Handle() func(w http.ResponseWriter, req *http.Request) {
	return func(w http.ResponseWriter, req *http.Request) {

		l.mux.Lock()
		if l.Plots != nil {
			_, err := l.Plots.WriteTo(w)
			if err != nil {
				panic(err)
			}
		}
		l.mux.Unlock()
	}

}
