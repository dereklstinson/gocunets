package tensor

import (
	"errors"
	"image"
	"image/color"
	"runtime"
	"sync"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/utils"
	gocudnn "github.com/dereklstinson/gocudnn"
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

//ToOneImageColor will return an image.Image of the volume in batch/neuron for the rows, and channels for the column
//Y and X represent how much padding of (hopefully black) there is between the channels and neurons
func (t *Volume) ToOneImageColor(h *cudnn.Handler, X, Y int) (image.Image, error) {
	images, err := t.ToImagesColor(h)
	if err != nil {
		return nil, err
	}
	return ToOneImage(images, X, Y), nil
}

//ToImagesColor makes a 2d array of images. Negative will be Green. Positive will be purple it returns a double array of image.Image. Only Float32 support right now
func (t *Volume) ToImagesColor(h *cudnn.Handler) ([][]image.Image, error) {
	dims := t.current.tD.Dims()
	vol := utils.FindVolumeInt32(dims, nil)
	slice := make([]float32, vol)
	err := t.FillSlice(h, slice)
	if err != nil {
		return nil, err
	}
	return convertcolor(slice, dims, t.frmt)
}

func convertcolor(data []float32, dims []int32, frmt gocudnn.TensorFormat) ([][]image.Image, error) {
	if len(dims) > 4 {
		return nil, errors.New("Dims of 4 only supported")
	}

	var tf gocudnn.TensorFormat

	var conv []int

	//	fmt.Println("ToImages:convert:Length of Slice", len(slice))

	//	fmt.Println("Converting Slice to uint8")
	conv = colornormal(data)

	var maxy int
	var maxx int
	var chans int          //these are usually the channels in the image or number of feature maps per neuron
	number := int(dims[0]) // usually number of samples in a batch, or the number of neurons in a layer
	imgs := make([][]image.Image, number)
	//fmt.Println("making image.Image")
	switch frmt {
	case tf.NCHW():

		chans = int(dims[1])
		maxy = int(dims[2])
		maxx = int(dims[3])
		var wg sync.WaitGroup
		maxthreads := runtime.NumCPU()
		counter := 0
		for i := 0; i < number; i++ {
			imgs[i] = make([]image.Image, chans)
			for j := 0; j < chans; j++ {
				wg.Add(1)
				go func(i, j int) {
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
							c := color.RGBA{R: R, G: G, B: B, A: 255}

							img.Set(l, k, c)

						}
					}
					imgs[i][j] = img
					wg.Done()
				}(i, j)
				if (counter)%maxthreads == maxthreads-1 {
					wg.Wait()

				}
				counter++
			}
		}
		wg.Wait()
		return imgs, nil
	case tf.NHWC():
		chans = int(dims[3])
		maxy = int(dims[1])
		maxx = int(dims[3])
		var wg sync.WaitGroup
		maxthreads := runtime.NumCPU()
		for i := 0; i < number; i++ {
			imgs[i] = make([]image.Image, chans)
			for j := 0; j < chans; j++ {
				wg.Add(1)
				go func(i, j int) {

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
					wg.Done()
				}(i, j)
				if (i*chans+j)%maxthreads == maxthreads-1 {
					wg.Wait()
				}
			}
		}
		wg.Wait()
		return imgs, nil
	case tf.NCHWvectC():
		chans = int(dims[1])
		maxy = int(dims[2])
		maxx = int(dims[3])
		var wg sync.WaitGroup
		maxthreads := runtime.NumCPU()

		for i := 0; i < number; i++ {
			imgs[i] = make([]image.Image, chans)
			for j := 0; j < chans; j++ {
				wg.Add(1)
				go func(i, j int) {
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
					wg.Done()
				}(i, j)
				if (i*chans+j)%maxthreads == maxthreads-1 {
					wg.Wait()
				}
			}
		}
		wg.Wait()
		return imgs, nil
	}

	return imgs, nil
}
