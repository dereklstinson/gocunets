package tensor

import (
	"errors"
	"fmt"
	"image"
	"image/color"

	"github.com/dereklstinson/GoCuNets/cudnn"
)

func findabsolutemaxfloat32(params []float32) float32 {
	max := float32(-1)
	var val float32
	for i := range params {
		if params[i] < 0 {
			val = -params[i]
		} else {
			val = params[i]
		}
		if val > max {
			max = val
		}
	}
	return max
}
func colornormal(params []float32) []int {
	absmax := findabsolutemaxfloat32(params)
	values := make([]int, len(params))
	//two55 := float32(255)
	for i := range params {
		values[i] = int((params[i] * 255) / absmax)
	}
	return values
}

//ToImagesColor makes a 2d array of images. Negative will be Green. Positive will be purple it returns a double array of image.Image. Only Float32 support right now
func (t *Volume) ToImagesColor() ([][]image.Image, error) {
	return t.convertColor()
}
func (t *Volume) convertColor() ([][]image.Image, error) {
	frmt, dtype, dims, err := t.Properties()
	if len(dims) > 4 {
		return nil, errors.New("Dims of 4 only supported")
	}

	var tf cudnn.TensorFormatFlag
	var dt cudnn.DataTypeFlag
	if err != nil {
		return nil, err
	}
	var conv []int
	switch dtype {
	case dt.Double():
		return nil, errors.New("Not supported")
	case dt.Float():
		slice := make([]float32, arraysizefromdims(dims))
		fmt.Println("ToImages:convert:Length of Slice", len(slice))
		err = t.memgpu.FillSlice(slice)
		if err != nil {
			return nil, err
		}
		fmt.Println("Converting Slice to uint8")
		conv = colornormal(slice)
		if err != nil {
			return nil, err
		}

	case dt.Int32():
		return nil, errors.New("Not supported")
	case dt.Int8():
		return nil, errors.New("Conversion for Int8 Not supported")
	case dt.UInt8():
		return nil, errors.New("Not supported")
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
						var (
							R uint8
							G uint8
							B uint8
						)
						if pix < 0 {
							R = uint8(255 + pix)
							G = uint8(255)
							B = uint8(255 + pix)
						} else {
							R = uint8(127 + (pix / 2))
							G = uint8(pix)
							B = uint8(127 + (pix / 2))
						}
						img.Set(l, k, color.RGBA{R, G, B, 255})
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
						var (
							R uint8
							G uint8
							B uint8
						)
						if pix < 0 {
							R = uint8(255 + pix)
							G = uint8(255)
							B = uint8(255 + pix)
						} else {
							R = uint8(127 + (pix / 2))
							G = uint8(pix)
							B = uint8(127 + (pix / 2))
						}
						img.Set(l, k, color.RGBA{R, G, B, 255})
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
						var (
							R uint8
							G uint8
							B uint8
						)
						if pix < 0 {
							R = uint8(255 + pix)
							G = uint8(255)
							B = uint8(255 + pix)
						} else {
							R = uint8(127 + (pix / 2))
							G = uint8(pix)
							B = uint8(127 + (pix / 2))
						}
						img.Set(l, k, color.RGBA{R, G, B, 255})
					}
				}
				imgs[i][j] = img
			}
		}
	}

	return imgs, nil
}
