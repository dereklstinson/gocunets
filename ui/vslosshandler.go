package ui

import (
	"fmt"
	"io"
	"net/http"
	"sync"

	"github.com/dereklstinson/GoCuNets/ui/plot"
	"gonum.org/v1/plot/plotter"
)

//VSLossHandler draws a vs chart for loss.  You might want to put several things in it like Training Loss Vs Testing Loss.
type VSLossHandler struct {
	Plots   io.WriterTo
	ceiling float32
	title   string
	xaxis   string
	yaxis   string
	h, w    int
	data    []plot.LabeledData
	index   int
	epoc    bool
	mux     sync.Mutex
}

//LabelFloat is used to pass data through channels to the VSLossHandler
type LabelFloat struct {
	Label string
	Data  float32
}

//NewVSLossHandle Makes a VSLoss Handle.  If epoc is true it will label the x axis as epoc. Otherwise it will label it batch.
// Just make sure you are passing the right amount of data through the LossData Channel.  So, the axis is labeled correctly.
// Y axis will be labeled Loss.
func NewVSLossHandle(title string, ceiling float32, LossData <-chan []LabelFloat, epoc bool, epocsize, batchskip int, labels ...string) (*VSLossHandler, []LabelFloat) {
	var xaxis string
	if epoc == true {
		xaxis = "Epocs"
	} else {
		xaxis = fmt.Sprintf("Every %d Batch", batchskip)
	}
	numberofplots := len(labels)

	x := &VSLossHandler{
		epoc:    epoc,
		ceiling: ceiling,
		title:   title,
		xaxis:   xaxis,
		yaxis:   "Loss",
		h:       6,
		w:       15,
		data:    makeinitializedlabeldata(numberofplots, labels, epocsize/batchskip),
	}

	go x.runchannel(LossData)
	lblflt := make([]LabelFloat, numberofplots)
	for i := range lblflt {
		lblflt[i].Label = labels[i]
	}
	return x, lblflt
}
func makeinitializedlabeldata(numberofplots int, labels []string, lengthofplots int) []plot.LabeledData {
	data := make([]plot.LabeledData, numberofplots)
	for i := range data {
		data[i].Label = labels[i]
		data[i].Data = make(plotter.XYs, lengthofplots)
		for j := range data[i].Data {
			data[i].Data[j].X = float64(j)
			data[i].Data[j].Y = float64(0)
		}
	}
	return data
}

func (l *VSLossHandler) runchannel(LossData <-chan []LabelFloat) {

	//	var err error
	for array := range LossData {
		var wg sync.WaitGroup
		l.mux.Lock()
		for i := range array {
			wg.Add(1)
			go func(i int) {

				placeandshiftback(l.data[i], array[i].Data)
				wg.Done()
			}(i)

		}
		wg.Wait()
		l.mux.Unlock()

	}
}

//ChangeHW changes the Height and Width of the Plot default is h= 6 cm, and w = 12 cm
func (l *VSLossHandler) ChangeHW(h, w int) {
	l.mux.Lock()
	l.h, l.w = h, w
	l.mux.Unlock()

}

//Handle is the func used for the web handler
func (l *VSLossHandler) Handle() func(w http.ResponseWriter, req *http.Request) {
	return func(w http.ResponseWriter, req *http.Request) {

		l.mux.Lock()
		var err error
		l.Plots, err = plot.Verses2(l.title, l.xaxis, l.yaxis, l.h, l.w, l.data)
		if err != nil {
			fmt.Println(err)
		}
		if l.Plots != nil {
			_, err = l.Plots.WriteTo(w)
			if err != nil {
				fmt.Println(err)
			}
		}
		l.mux.Unlock()

	}

}

func placeandshiftback(a plot.LabeledData, in float32) {
	b := 0.0
	input := float64(in)

	for i := a.Data.Len() - 1; i >= 0; i-- {
		b = a.Data[i].Y
		a.Data[i].Y = input
		input = b
	}
}
