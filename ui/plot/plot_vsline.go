package plot

import (
	"io"
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

//Verses does a verses line plot with the data passed.
//title,xaxis,yaxis are the labels for the plot image
//h,w are the size of the image
//returns a WriteTo.  I did this because I thought the user might want to store an array of them
func Verses(title, xaxis, yaxis string, h, w int, data ...LabeledData) (io.WriterTo, error) {
	rand.Seed(int64(20))

	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.X.Min = 0
	p.Y.Min = 0
	p.Title.Text = title
	p.X.Label.Text = xaxis
	p.Y.Label.Text = yaxis
	for i := range data {
		err = plotutil.AddLines(p, data[i].Label, data[i].Data)
		if err != nil {
			panic(err)
		}
	}
	return p.WriterTo(vg.Length(h)*vg.Centimeter, vg.Length(w)*vg.Centimeter, "jpg")

}

//Verses2 does a verses line plot with the data passed.
//title,xaxis,yaxis are the labels for the plot image
//h,w are the size of the image
//returns a WriteTo.  I did this because I thought the user might want to store an array of them
//Verse2 is the same as verses accept it takes a slice of LabeledData instead of a bunch of arguments of LabeledData
func Verses2(title, xaxis, yaxis string, h, w int, data []LabeledData) (io.WriterTo, error) {
	rand.Seed(int64(20))

	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.X.Min = 0
	p.Y.Min = 0
	p.Title.Text = title
	p.X.Label.Text = xaxis
	p.Y.Label.Text = yaxis

	for i := range data {

		err = plotutil.AddLines(p,
			data[i].Label, data[i].Data)
		if err != nil {
			panic(err)
		}
	}
	return p.WriterTo(vg.Length(w)*vg.Centimeter, vg.Length(h)*vg.Centimeter, "jpg")

}

//Verses3 does a verses line plot with the data passed.
//title,xaxis,yaxis are the labels for the plot image
//h,w are the size of the image
//returns a WriteTo.  This will allow up to 12 LabelData to have individual colors
func Verses3(title, xaxis, yaxis string, h, w int, data []LabeledData) (io.WriterTo, error) {
	rand.Seed(int64(20))

	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.X.Min = 0
	p.Y.Min = 0
	p.Title.Text = title
	p.X.Label.Text = xaxis
	p.Y.Label.Text = yaxis

	switch len(data) {
	case 1:
		err = plotutil.AddLines(p,
			data[0].Label, data[0].Data)
		if err != nil {
			panic(err)
		}
	case 2:
		err = plotutil.AddLines(p,
			data[0].Label, data[0].Data,
			data[1].Label, data[1].Data)
		if err != nil {
			panic(err)
		}
	case 3:
		err = plotutil.AddLines(p,
			data[0].Label, data[0].Data,
			data[1].Label, data[1].Data,
			data[2].Label, data[2].Data,
		)
		if err != nil {
			panic(err)
		}
	case 4:
		err = plotutil.AddLines(p,
			data[0].Label, data[0].Data,
			data[1].Label, data[1].Data,
			data[2].Label, data[2].Data,
			data[3].Label, data[3].Data,
		)
		if err != nil {
			panic(err)
		}
	case 5:
		err = plotutil.AddLines(p,
			data[0].Label, data[0].Data,
			data[1].Label, data[1].Data,
			data[2].Label, data[2].Data,
			data[3].Label, data[3].Data,
			data[4].Label, data[4].Data,
		)
		if err != nil {
			panic(err)
		}
	case 6:
		err = plotutil.AddLines(p,
			data[0].Label, data[0].Data,
			data[1].Label, data[1].Data,
			data[2].Label, data[2].Data,
			data[3].Label, data[3].Data,
			data[4].Label, data[4].Data,
			data[5].Label, data[5].Data,
		)
		if err != nil {
			panic(err)
		}
	case 7:
		err = plotutil.AddLines(p,
			data[0].Label, data[0].Data,
			data[1].Label, data[1].Data,
			data[2].Label, data[2].Data,
			data[3].Label, data[3].Data,
			data[4].Label, data[4].Data,
			data[5].Label, data[5].Data,
			data[6].Label, data[6].Data,
		)
		if err != nil {
			panic(err)
		}
	case 8:
		err = plotutil.AddLines(p,
			data[0].Label, data[0].Data,
			data[1].Label, data[1].Data,
			data[2].Label, data[2].Data,
			data[3].Label, data[3].Data,
			data[4].Label, data[4].Data,
			data[5].Label, data[5].Data,
			data[6].Label, data[6].Data,
			data[7].Label, data[7].Data,
		)
		if err != nil {
			panic(err)
		}
	case 9:
		err = plotutil.AddLines(p,
			data[0].Label, data[0].Data,
			data[1].Label, data[1].Data,
			data[2].Label, data[2].Data,
			data[3].Label, data[3].Data,
			data[4].Label, data[4].Data,
			data[5].Label, data[5].Data,
			data[6].Label, data[6].Data,
			data[7].Label, data[7].Data,
			data[8].Label, data[8].Data,
		)
		if err != nil {
			panic(err)
		}
	case 10:
		err = plotutil.AddLines(p,
			data[0].Label, data[0].Data,
			data[1].Label, data[1].Data,
			data[2].Label, data[2].Data,
			data[3].Label, data[3].Data,
			data[4].Label, data[4].Data,
			data[5].Label, data[5].Data,
			data[6].Label, data[6].Data,
			data[7].Label, data[7].Data,
			data[8].Label, data[8].Data,
			data[9].Label, data[9].Data,
		)
		if err != nil {
			panic(err)
		}
	case 11:
		err = plotutil.AddLines(p,
			data[0].Label, data[0].Data,
			data[1].Label, data[1].Data,
			data[2].Label, data[2].Data,
			data[3].Label, data[3].Data,
			data[4].Label, data[4].Data,
			data[5].Label, data[5].Data,
			data[6].Label, data[6].Data,
			data[7].Label, data[7].Data,
			data[8].Label, data[8].Data,
			data[9].Label, data[9].Data,
			data[10].Label, data[10].Data,
		)
		if err != nil {
			panic(err)
		}
	case 12:
		err = plotutil.AddLines(p,
			data[0].Label, data[0].Data,
			data[1].Label, data[1].Data,
			data[2].Label, data[2].Data,
			data[3].Label, data[3].Data,
			data[4].Label, data[4].Data,
			data[5].Label, data[5].Data,
			data[6].Label, data[6].Data,
			data[7].Label, data[7].Data,
			data[8].Label, data[8].Data,
			data[9].Label, data[9].Data,
			data[10].Label, data[10].Data,
			data[11].Label, data[11].Data,
		)
		if err != nil {
			panic(err)
		}
	default:
		panic("Length of LabeledData should be 1<=LabeledData<=12 ")

	}

	return p.WriterTo(vg.Length(w)*vg.Centimeter, vg.Length(h)*vg.Centimeter, "jpg")

}
