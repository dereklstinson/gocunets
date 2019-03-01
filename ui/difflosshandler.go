package ui

import (
	"io"
	"net/http"
	"sync"

	"github.com/dereklstinson/GoCuNets/ui/plot"
)

//DifLossHandler draws a vs chart for loss.  You might want to put several things in it like Training Loss Vs Testing Loss.
type DifLossHandler struct {
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

//NewDifLossHandler Makes a VSLoss Handle.  If epoc is true it will label the x axis as epoc. Otherwise it will label it batch.
// Just make sure you are passing the right amount of data through the LossData Channel.  So, the axis is labeled correctly.
// Y axis will be labeled Loss.
func NewDifLossHandler(title string, LossData <-chan []LabelFloat2, epoc bool, plotlengths int, labels ...string) (*DifLossHandler, []LabelFloat2) {
	var xaxis string
	if epoc == true {
		xaxis = "Epocs"
	} else {
		xaxis = "Batch"
	}
	numberofplots := len(labels)
	y := make([][]float32, numberofplots)
	data := make([]plot.LabeledData, numberofplots)
	x := &DifLossHandler{
		epoc:         epoc,
		title:        title,
		xaxis:        xaxis,
		yaxis:        "Loss",
		h:            8,
		w:            20,
		originaldata: y,
		data:         data,
	}

	go x.runchannel(LossData)
	lblflt := make([]LabelFloat2, numberofplots)
	for i := range lblflt {
		lblflt[i].Data = make([]float32, plotlengths)
		lblflt[i].Label = labels[i]
	}
	return x, lblflt
}

//LabelFloat2 wraps a label (string) and data (float array)
type LabelFloat2 struct {
	Label string
	Data  []float32
}

func (l *DifLossHandler) runchannel(LossData <-chan []LabelFloat2) {

	var err error
	for array := range LossData {
		if len(array) != 2 {
			panic("Length of array needs to be 2")
		}
		for i := range array {
			l.originaldata[i] = append(l.originaldata[i], array[i].Data...)
			l.data[i], err = plot.NewLabeledData(array[i].Label, l.originaldata[i])
		}
		l.mux.Lock()
		l.Plots, err = plot.Difference(l.title, l.xaxis, l.yaxis, l.h, l.w, l.data[0], l.data[1])
		l.mux.Unlock()
		if err != nil {
			panic(err)
		}
	}
}

//ChangeHW changes the Height and Width of the Plot default is h= 6 cm, and w = 12 cm
func (l *DifLossHandler) ChangeHW(h, w int) {
	l.mux.Lock()
	l.h, l.w = h, w
	l.mux.Unlock()

}

//Handle is  used for the web handler
func (l *DifLossHandler) Handle() func(w http.ResponseWriter, req *http.Request) {
	return func(w http.ResponseWriter, req *http.Request) {
		l.mux.Lock()
		_, err := l.Plots.WriteTo(w)
		if err != nil {
			panic(err)
		}
		l.mux.Unlock()
	}

}
