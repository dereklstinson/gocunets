package plot

import (
	"errors"
	"io"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

//LabeledData holdes the label and the data for the plot
type LabeledData struct {
	Label string
	Data  plotter.XYs
}

//NewLabeledData will maake a labeled data.  data supported is []float32, []float64, and plotter.XYs.
func NewLabeledData(label string, data interface{}) (LabeledData, error) {

	switch x := data.(type) {
	case plotter.XYs:
		return LabeledData{
			Label: label,
			Data:  x,
		}, nil
	case []float64:
		pts := make(plotter.XYs, len(x))
		for i := range x {
			pts[i].X = float64(i)
			pts[i].Y = x[i]
		}
		return LabeledData{
			Label: label,
			Data:  pts,
		}, nil
	case []float32:
		pts := make(plotter.XYs, len(x))
		for i := range x {
			pts[i].X = float64(i)
			pts[i].Y = float64(x[i])
		}
		return LabeledData{
			Label: label,
			Data:  pts,
		}, nil
	}
	return LabeledData{}, errors.New("Not Supported data")
}

//Difference will subtract that datapoints of the Y axis of B from A.  It will also add to the plot  "A.Label - B.Label"
//A and B need to have the same number of elements.  This only works on discrete n arrays so x=i intigers
//It will return an io.WriterTo and an error
func Difference(title, xaxis, yaxis string, h, w int, A, B LabeledData) (io.WriterTo, error) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = title
	p.X.Label.Text = xaxis
	p.Y.Label.Text = yaxis
	pts := make(plotter.XYs, len(A.Data))
	for i := range A.Data {
		pts[i].Y = A.Data[i].Y - B.Data[i].Y
		pts[i].X = float64(i)
	}
	//	plotter.NewHistogram()
	err = plotutil.AddLines(p, A.Label+"-"+B.Label, pts)
	return p.WriterTo(vg.Length(h)*vg.Centimeter, vg.Length(w)*vg.Centimeter, "jpg")

}
