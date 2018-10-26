package imaging

//build
import (
	"errors"
	"image"
	"image/color"

	"github.com/dereklstinson/GoCuNets/gocudnn/reshapes"
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Imager takes tensors and to the best its ability turn it into an image.Image
type Imager struct {
	shaper *reshapes.Ops
}

//MakeImager makes an imager
func MakeImager(handle *gocudnn.XHandle) (*Imager, error) {
	shpr, err := reshapes.Stage(handle)
	return &Imager{
		shaper: shpr,
	}, err
}

//TileBatches will take the batches and lay place them withing the HWC space like tiles.
//Channel dim is limited to 1-4. If c=1: [r,r,r,255]; If c=2: [r,g,avg(r,g),255]; c=3: [r,g,b,255]; c=4: [r,g,b,a];
func (im *Imager) TileBatches(handle *gocudnn.XHandle, x *layers.IO, h, w int) (image.Image, error) {

	frmt, dtype, dims, managed, err := im.shaper.GetB2SOutputProperties(handle, x.T(), []int32{int32(h), int32(w)})
	if err != nil {
		return nil, err
	}

	output, err := tensor.Build(frmt, dtype, dims, managed)
	if err != nil {
		return nil, err
	}
	defer output.Destroy()
	err = im.shaper.S2BBackward(handle, output, x.T())
	if err != nil {
		return nil, err
	}
	var dflg gocudnn.DataTypeFlag
	vol := utils.FindVolumeInt32(dims)
	var z []float32
	switch dtype {
	case dflg.Double():
		y := make([]float64, vol)
		z = converttofloat32(y)
	case dflg.Float():
		y := make([]float32, vol)
		z = converttofloat32(y)
	case dflg.Int32():
		y := make([]int32, vol)
		z = converttofloat32(y)
	case dflg.Int8():
		y := make([]int8, vol)
		z = converttofloat32(y)
	case dflg.UInt8():
		y := make([]uint8, vol)
		z = converttofloat32(y)
	}
	if z == nil {
		return nil, errors.New("Unsupported Datatype")
	}
	output.Memer().FillSlice(z)
	return makeimage(z, dims, frmt)

}

func converttofloat32(input interface{}) []float32 {
	switch x := input.(type) {
	case []float32:
		return x
	case []int:

		array := make([]float32, len(x))
		for i := range x {
			array[i] = float32(x[i])
		}
		return array
	case []int32:
		array := make([]float32, len(x))
		for i := range x {
			array[i] = float32(x[i])
		}
		return array
	case []float64:
		array := make([]float32, len(x))
		for i := range x {
			array[i] = float32(x[i])
		}
		return array
	case []int8:
		array := make([]float32, len(x))
		for i := range x {
			array[i] = float32(x[i])
		}
		return array
	case []int64:

		array := make([]float32, len(x))
		for i := range x {
			array[i] = float32(x[i])
		}
		return array
	}
	return nil
}

func makeimage(data []float32, dims []int32, frmt gocudnn.TensorFormat) (image.Image, error) {
	unNormalize(data)
	if len(data) != int(utils.FindVolumeInt32(dims)) {
		return nil, errors.New("Volume Size doesn't Match intput size")
	}
	var fflag gocudnn.TensorFormatFlag
	var h int
	var w int
	var c int
	a := uint8(255)
	switch frmt {

	case fflag.NCHW():
		h = int(dims[2])
		w = int(dims[3])
		c = int(dims[1])
		rect1 := image.Rect(0, 0, w, h)
		img1 := image.NewRGBA(rect1)
		switch c {
		case 4:
			for i := 0; i < h; i++ {
				for j := 0; j < w; j++ {
					r1 := uint8(data[(0*h*w)+(i*w)+j])
					g1 := uint8(data[(1*w*c)+(i*w)+j])
					b1 := uint8(data[(2*w*c)+(i*w)+j])
					a := uint8(data[(3*w*c)+(i*w)+j])
					img1.Set(j, i, color.RGBA{r1, g1, b1, a})
				}
			}

		case 3:
			for i := 0; i < h; i++ {
				for j := 0; j < w; j++ {
					r1 := uint8(data[(0*h*w)+(i*w)+j])
					g1 := uint8(data[(1*w*c)+(i*w)+j])
					b1 := uint8(data[(2*w*c)+(i*w)+j])
					img1.Set(j, i, color.RGBA{r1, g1, b1, a})
				}
			}
		case 2:
			for i := 0; i < h; i++ {
				for j := 0; j < w; j++ {
					r1 := uint8(data[(0*h*w)+(i*w)+j])
					g1 := uint8(data[(1*w*c)+(i*w)+j])
					b1 := (r1 + g1) / 2
					img1.Set(j, i, color.RGBA{r1, g1, b1, a})
				}
			}

		case 1:
			for i := 0; i < h; i++ {
				for j := 0; j < w; j++ {
					r1 := uint8(data[(0*h*w)+(i*w)+j])
					img1.Set(j, i, color.RGBA{r1, r1, r1, a})
				}
			}

		default:
			return nil, errors.New("Not Supported channel amount")
		}

		return img1, nil

	case fflag.NHWC():
		h = int(dims[1])
		w = int(dims[2])
		c = int(dims[3])
		rect1 := image.Rect(0, 0, w, h)
		img1 := image.NewRGBA(rect1)
		switch c {
		case 4:
			for i := 0; i < h; i++ {
				for j := 0; j < w; j++ {
					r1 := uint8(data[(i*w*c)+(j*c)+0])
					g1 := uint8(data[(i*w*c)+(j*c)+1])
					b1 := uint8(data[(i*w*c)+(j*c)+2])
					a = uint8(data[(i*w*c)+(j*c)+3])
					img1.Set(j, i, color.RGBA{r1, g1, b1, a})
				}
			}
		case 3:

			for i := 0; i < h; i++ {
				for j := 0; j < w; j++ {
					r1 := uint8(data[(i*w*c)+(j*c)+0])
					g1 := uint8(data[(i*w*c)+(j*c)+1])
					b1 := uint8(data[(i*w*c)+(j*c)+2])
					img1.Set(j, i, color.RGBA{r1, g1, b1, a})
				}
			}
		case 2:
			for i := 0; i < h; i++ {
				for j := 0; j < w; j++ {
					r1 := uint8(data[(i*w*c)+(j*c)+0])
					g1 := uint8(data[(i*w*c)+(j*c)+1])
					b1 := (r1 + g1) / 2
					img1.Set(j, i, color.RGBA{r1, g1, b1, a})
				}
			}
		case 1:
			for i := 0; i < h; i++ {
				for j := 0; j < w; j++ {
					r1 := uint8(data[(i*w*c)+(j*c)+0])
					img1.Set(j, i, color.RGBA{r1, r1, r1, a})
				}
			}

		default:
			return nil, errors.New("Not Supported channel amount")
		}

		return img1, nil
	default:
		return nil, errors.New("Format Not Supported")

	}

}

func minvalue(data []float32) float32 {
	var min float32
	min = float32(99999999)
	for i := 0; i < len(data); i++ {
		if data[i] < min {
			min = data[i]
		}
	}
	return min
}
func maxvalue(data []float32) float32 {
	var max float32
	max = float32(-99999999)
	for i := 0; i < len(data); i++ {
		if data[i] > max {
			max = data[i]
		}
	}
	return max
}
func multiplydata(data []float32, value float32) {
	for i := 0; i < len(data); i++ {
		data[i] *= value
	}

}
func unNormalize(data []float32) {

	min := minvalue(data)
	adddata(data, -min)
	max := maxvalue(data)
	multiplydata(data, float32(255)/max)
}
func adddata(data []float32, value float32) {
	for i := 0; i < len(data); i++ {
		data[i] += value
	}
}
