package roman

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/dereklstinson/GoCuNets/utils"
	"github.com/nfnt/resize"
)

//Roman is some roman numerals 1 through 9 (I through IX) with the added character N.  N == 0
type Roman struct {
	Data   []float32
	Number int
}

//GetRoman will get the roman images and convert it to a Roman struct.
//The function will panic on an error.
func GetRoman(folder string) []Roman {
	imgs := getromanimages(folder)

	imgsize := 28 * 28
	pixels := make([]float32, 0, imgsize*10)
	for i := range imgs {
		imgs[i] = make28by28(imgs[i])
		pixels = append(pixels, makepixels(imgs[i])...)
	}

	//	max := utils.FindMax(pixels)
	//	utils.DivideAll(pixels, max)
	//	fmt.Println("Max", max)
	//average := utils.FindAvg(pixels)
	//fmt.Println("Avg:", average)
	//	utils.AddAll(pixels, -average)
	//average = utils.FindAvg(pixels)
	//fmt.Println("NewAvg:", average)
	label := 0
	romans := make([]Roman, 0)
	/*
		for i := range pixels {
			fmt.Println("i:", i, "Pixel value", pixels[i])
		}
	*/
	for i := 0; i < len(pixels); i += imgsize {
		label = i / imgsize
		romans = append(romans, makeRoman(pixels[i:i+imgsize], label))

	}
	return romans
}

//Copy makes a copy of Roman
func (r *Roman) Copy() (cpy Roman) {
	cpy.Data = make([]float32, len(r.Data))
	cpy.Number = r.Number
	for i := range r.Data {
		cpy.Data[i] = r.Data[i]
	}
	return cpy

}
func EncodeSoftmaxPerPixel(data []Roman) []Roman {

	for i := range data {
		offset := len(data[i].Data)
		newdata := make([]float32, len(data[i].Data)*2)
		for j := range data[i].Data {
			if data[i].Data[j] < 128 {
				newdata[j] = 0
				newdata[offset+j] = 1
			} else {
				newdata[j] = 1
				newdata[offset+j] = 0
			}
		}
		data[i].Data = newdata
	}
	return data
}

//Normalize Normalizes the data
func Normalize(data []Roman, avg float32) []Roman {
	for i := range data {
		utils.DivideAll(data[i].Data, float32(255))
		utils.AddAll(data[i].Data, -avg)
	}
	return data
}
func makeRoman(data []float32, label int) Roman {
	var roman Roman
	roman.Number = label
	roman.Data = make([]float32, 28*28)
	for i := range data {
		roman.Data[i] = data[i]
	}
	return roman
}
func cherr(err error) {
	if err != nil {
		panic(err)
	}
}
func makepixels(im image.Image) []float32 {
	x := im.Bounds().Max.X
	y := im.Bounds().Max.Y
	pixels := make([]float32, x*y)
	for i := 0; i < y; i++ {
		for k := 0; k < x; k++ {
			r, g, b, _ := im.At(k, i).RGBA()
			avg := (r + g + b) / 3
			avg = 255 - ((avg * 255) / 65535)
			pixels[i*x+k] = float32(avg)
		}

	}
	return pixels
}

func make28by28(im image.Image) image.Image {
	return resize.Resize(28, 28, im, resize.NearestNeighbor)
}

func getromanimages(folder string) []image.Image {
	images := make([]image.Image, 10)

	filepath.Walk(folder, func(path string, info os.FileInfo, err error) error {
		if info.IsDir() {
			return nil
		}
		nosuffix := strings.TrimSuffix(path, ".png")
		numberlabel := strings.TrimPrefix(nosuffix, folder)
		numbint, err := strconv.Atoi(numberlabel)
		if err != nil {
			return nil
		}
		roman, err := loadimage(path)
		images[numbint] = roman
		if err != nil {
			return err
		}
		return nil
	})
	return images
}

func loadimage(path string) (image.Image, error) {
	var img image.Image
	file, err := os.Open(path)
	if err != nil {
		fmt.Println("OpenFile:", path)
		return nil, err
	}
	defer file.Close()

	if strings.HasSuffix(path, ".jpg") == true {
		img, err = jpeg.Decode(file)
		if err != nil {
			fmt.Println("JPG: ", path, ".jpg")
			return nil, err
		}

	} else {

		img, err = png.Decode(file)

		if err != nil {
			fmt.Println("PNG: ", path, ".png")
			return nil, err
		}

	}

	return img, nil
}
