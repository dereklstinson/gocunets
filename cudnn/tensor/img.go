package tensor

/*
import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"strconv"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

func arraysizefromdims(dims []int32) int {
	mult := 1
	for i := 0; i < len(dims); i++ {
		mult *= int(dims[i])
	}
	return mult
}

func unnormfloat32(input []float32) {
	imin := float32(999999999.0)
	imax := -float32(99999999.0)
	for i := 0; i < len(input); i++ {
		if imin > input[i] {
			imin = input[i]
		}
		if imax < input[i] {
			imax = input[i]
		}
	}

	irange := imax - imin
	ratio := float32(255) / irange
	for i := 0; i < len(input); i++ {
		input[i] -= imin
		input[i] *= ratio

	}

}
func unnormfloat64(input []float64) {
	imin := float64(999999999.0)
	imax := -float64(99999999.0)
	for i := 0; i < len(input); i++ {
		if imin > input[i] {
			imin = input[i]
		}
		if imax < input[i] {
			imax = input[i]
		}
	}

	irange := imax - imin
	ratio := float64(255) / irange
	for i := 0; i < len(input); i++ {
		input[i] -= imin
		input[i] *= ratio

	}
}

func unnormint32(input []int32) {
	imin := float32(999999999.0)
	imax := -float32(99999999.0)
	for i := 0; i < len(input); i++ {
		if imin > float32(input[i]) {
			imin = float32(input[i])
		}
		if imax < float32(input[i]) {
			imax = float32(input[i])
		}
	}

	irange := imax - imin
	ratio := float32(255) / irange
	for i := 0; i < len(input); i++ {
		input[i] -= int32(imin)
		input[i] = int32(float32(input[i]) * ratio)

	}

}

func unnormalize(input interface{}) error {
	switch x := input.(type) {
	case []float32:
		unnormfloat32(x)
		return nil
	case []int32:
		unnormint32(x)
		return nil
	case []float64:
		unnormfloat64(x)
		return nil
	case []uint8:
		//do nothing
		return nil
	default:
		return errors.New("Current type not supported in reformating to image")

	}

}
func touint8(input interface{}) ([]uint8, error) {
	switch x := input.(type) {

	case []float32:

		unnormalize(x)
		length := len(x)
		array := make([]uint8, length)
		for i := 0; i < length; i++ {
			array[i] = uint8(x[i])
		}
		return array, nil
	case []int32:

		unnormalize(x)
		length := len(x)
		array := make([]uint8, length)
		for i := 0; i < length; i++ {
			array[i] = uint8(x[i])
		}
		return array, nil
	case []float64:

		unnormalize(x)
		length := len(x)
		array := make([]uint8, length)
		for i := 0; i < length; i++ {
			array[i] = uint8(x[i])
		}
		return array, nil
	case []uint8:
		return x, nil
	default:
		return nil, errors.New("type passed not supported to uint8conversion")

	}
}

//MakeJPG will make a JPG file for the image passed it can be sepperated by neurons and the such
func MakeJPG(folder, subfldr string, index int, img image.Image) error {
	dir := folder + "/" + subfldr + "/"
	err := os.MkdirAll(dir, os.ModePerm)
	if err != nil {
		return err
	}
	newfile, err := os.Create(dir + strconv.Itoa(index) + ".jpg")
	if err != nil {
		return err
	}
	defer newfile.Close()

	return jpeg.Encode(newfile, img, nil)
}

//ToImages makes a 2d array of images.  If it is a filter then it will be listed by the images[x][y] x is the neurons. y is the feature maps for neuron x
//if it is a tensor then x is the batch. y is the channel. It doesn't matter what format was used
func (t *Volume) ToImages() ([][]image.Image, error) {
	return t.convert()
}
func (t *Volume) convert() ([][]image.Image, error) {
	frmt, dtype, dims, err := t.Properties()
	if len(dims) > 4 {
		return nil, errors.New("Dims of 4 only supported")
	}

	var tf gocudnn.TensorFormatFlag
	var dt gocudnn.DataTypeFlag
	if err != nil {
		return nil, err
	}
	var conv []uint8
	switch dtype {
	case dt.Double():
		slice := make([]float64, arraysizefromdims(dims))
		err = t.memgpu.FillSlice(slice)
		if err != nil {
			return nil, err
		}
		conv, err = touint8(slice)
		if err != nil {
			return nil, err
		}
	case dt.Float():
		slice := make([]float32, arraysizefromdims(dims))
		fmt.Println("ToImages:convert:Length of Slice", len(slice))
		err = t.memgpu.FillSlice(slice)
		if err != nil {
			return nil, err
		}
		fmt.Println("Converting Slice to uint8")
		conv, err = touint8(slice)
		if err != nil {
			return nil, err
		}

	case dt.Int32():
		slice := make([]int32, arraysizefromdims(dims))
		err = t.memgpu.FillSlice(slice)
		if err != nil {
			return nil, err
		}
		conv, err = touint8(slice)
		if err != nil {
			return nil, err
		}
	case dt.Int8():
		return nil, errors.New("Conversion for Int8 Not supported")
	case dt.UInt8():
		conv = make([]uint8, arraysizefromdims(dims))
		err = t.memgpu.FillSlice(conv)
		if err != nil {
			return nil, err
		}
	}

	var maxy int
	var maxx int
	var chans int          //these are usually the channels in the image or number of feature maps per neuron
	number := int(dims[0]) // usually number of samples in a batch, or the number of neurons in a layer
	imgs := make([][]image.Image, number)
	fmt.Println("making image.Image")
	switch frmt {
	case tf.NCHW():
		chans = int(dims[1])
		maxy = int(dims[2])
		maxx = int(dims[3])

		for i := 0; i < number; i++ {
			imgs[i] = make([]image.Image, chans)
			for j := 0; j < chans; j++ {
				var rect image.Rectangle
				rect.Min.X = 0
				rect.Min.Y = 0
				rect.Max.X = maxx
				rect.Max.Y = maxy
				img := image.NewRGBA(rect)
				for k := 0; k < maxy; k++ {
					for l := 0; l < maxx; l++ {
						pix := conv[(i*chans*maxy*maxx)+(j*maxy*maxx)+(k*maxx)+l]
						img.Set(j, i, color.RGBA{pix, pix, pix, 255})
					}
				}
				imgs[i][j] = img
			}
		}
	case tf.NHWC():
		chans = int(dims[3])
		maxy = int(dims[1])
		maxx = int(dims[3])
		for i := 0; i < number; i++ {
			imgs[i] = make([]image.Image, chans)
			for j := 0; j < chans; j++ {
				var rect image.Rectangle
				rect.Min.X = 0
				rect.Min.Y = 0
				rect.Max.X = maxx
				rect.Max.Y = maxy
				img := image.NewRGBA(rect)
				for k := 0; k < maxy; k++ {
					for l := 0; l < maxx; l++ {
						pix := conv[(i*chans*maxy*maxx)+(k*chans*maxx)+(k*chans)+j]
						img.Set(j, i, color.RGBA{pix, pix, pix, 255})
					}
				}
				imgs[i][j] = img
			}
		}

	case tf.NCHWvectC():
		chans = int(dims[1])
		maxy = int(dims[2])
		maxx = int(dims[3])

		for i := 0; i < number; i++ {
			imgs[i] = make([]image.Image, chans)
			for j := 0; j < chans; j++ {
				var rect image.Rectangle
				rect.Min.X = 0
				rect.Min.Y = 0
				rect.Max.X = maxx
				rect.Max.Y = maxy
				img := image.NewRGBA(rect)
				for k := 0; k < maxy; k++ {
					for l := 0; l < maxx; l++ {
						pix := conv[(i*chans*maxy*maxx)+(j*maxy*maxx)+(k*maxx)+l]
						img.Set(j, i, color.RGBA{pix, pix, pix, 255})
					}
				}
				imgs[i][j] = img
			}
		}
	}

	return imgs, nil
}
*/
