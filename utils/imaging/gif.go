package imaging

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"io"
	"sync"
)

//Giffer contains a gif and the delay.  It is used to append image.Images to a gif.
//So that it will eventually be able encode it to a writer
type Giffer struct {
	gify  gif.GIF
	delay int
}

//NewGiffer is an extention to the gif package from google.
//loopcount = -1 = one run, 0 = infinity, n = n loops
//delay is in 1/100 seconds
func NewGiffer(loopcount int, delay int) *Giffer {
	var gify gif.GIF
	gify.LoopCount = loopcount
	return &Giffer{
		gify:  gify,
		delay: delay,
	}
}

//MakeGrayGif will take an array of images and make a gif out if it
func (g *Giffer) MakeGrayGif(imgs []image.Image) {
	testcpucores := 16
	length := len(imgs)
	g.gify.Image = make([]*image.Paletted, length)
	g.gify.Delay = make([]int, length)
	var wg sync.WaitGroup
	fmt.Println("TOTAL IMAGES", length)
	for i := range g.gify.Image {
		wg.Add(1)
		go func(i int) {

			g.gify.Image[i] = topallet(imgs[i])
			g.gify.Delay[i] = g.delay

			wg.Done()
		}(i)
		if i%testcpucores == testcpucores-1 {

			wg.Wait()
			fmt.Println("Done", i, "images of", length)
		}

	}
	wg.Wait()
}

//Append appends an image to the Giffer
func (g *Giffer) Append(img image.Image) {
	g.gify.Image = append(g.gify.Image, topallet(img))
	g.gify.Delay = append(g.gify.Delay, g.delay)
}
func topallet(img image.Image) *image.Paletted {
	y := img.Bounds().Max.Y
	x := img.Bounds().Max.X
	p := grayPallet(uint8(1))
	palleted := image.NewPaletted(img.Bounds(), p)

	for i := 0; i < y; i++ {
		for j := 0; j < x; j++ {
			value := img.At(j, i)
			palleted.Set(j, i, value)
		}
	}
	return palleted
}

//Encode encodes the gif to the writer
func (g *Giffer) Encode(w io.Writer) error {
	return gif.EncodeAll(w, &g.gify)
}

func grayPallet(multiple uint8) color.Palette {

	length := uint8(255) / multiple
	if length < uint8(2) {
		panic(errors.New("need at least 2 shades"))
	}
	clrs := make([]color.Color, 0)
	for i := uint8(0); i < length; i++ {
		value := 255 - (i * multiple)
		if value < uint8(0) {
			return clrs
		}
		clrs = append(clrs, color.RGBA{value, value, value, 255})
	}

	return clrs
}
