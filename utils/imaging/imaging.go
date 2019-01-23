package imaging

//build
import (
	"errors"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/reshapes"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	"github.com/dereklstinson/GoCuNets/layers" //	"github.com/dereklstinson/GoCuNets/thirdparty/github.com/nfnt/resize"
	"github.com/dereklstinson/GoCuNets/utils"
	"github.com/nfnt/resize"
)

//Imager takes tensors and to the best its ability turn it into an image.Image
type Imager struct {
	shaper *reshapes.Ops
	cache  *tensor.Volume
}

//ImageV2 is a wrapper for an img.Image.  It is made so that it can be used in the Encoder interface.
type ImageV2 struct {
	img image.Image
	png bool
}

//UpgradeImagetoV2 upgrades an image.Image to an ImageV2 so it can be use the encoder interface
func UpgradeImagetoV2(img image.Image) *ImageV2 {
	return &ImageV2{
		img: img,
	}
}

//Bounds returns an image.Rectangle
func (i *ImageV2) Bounds() image.Rectangle {
	return i.img.Bounds()
}

//At returns the color.Color at x,y
func (i *ImageV2) At(x int, y int) color.Color {
	return i.img.At(x, y)
}

//ColorModel returns the colormodel for image
func (i *ImageV2) ColorModel() color.Model {
	return i.img.ColorModel()
}

//SetPNG will set the PNG flag.  if true then if will encode the image to png if false it will encode to jpg.
//Default is jpg
func (i *ImageV2) SetPNG(png bool) {
	i.png = png
}

//Encode will encode the imagev2 to a jpeg unless you set it to png with SetPNG
func (i *ImageV2) Encode(w io.Writer) error {
	if i.png == true {
		return png.Encode(w, i.img)
	}
	return jpeg.Encode(w, i.img, nil)
}

//MakeImager makes an imager
func MakeImager(handle *cudnn.Handler) (*Imager, error) {
	shpr, err := reshapes.Stage(handle)
	return &Imager{
		shaper: shpr,
	}, err
}

//ByBatches will return the images by batches if h xor w is zero then the ration will be kept. if both are zero then the ratio is not
func (im *Imager) ByBatches(handle *cudnn.Handler, x *layers.IO, h, w uint) ([]image.Image, error) {
	frmt, dtype, dims, err := x.Properties()
	if err != nil {
		return nil, err
	}
	var dflg cudnn.DataTypeFlag
	vol := utils.FindVolumeInt32(dims, nil)
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
	x.T().Memer().FillSlice(z)
	images := make([]image.Image, 0)
	batchvol := int(utils.FindVolumeInt32(dims[1:], nil))
	batchdims := []int32{1, dims[1], dims[2], dims[3]}
	for i := 0; i < int(dims[0]); i++ {
		bz := z[i*batchvol : (i+1)*batchvol]
		batchimage, err := makeimage(bz, batchdims, frmt)
		if err != nil {
			return nil, err
		}
		batchimage = resize.Resize(w, h, batchimage, resize.NearestNeighbor)
		images = append(images, batchimage)
	}

	return images, nil
}

//TileBatchesXdX will take the batches and lay place them withing the HWC space like tiles.It will do both of the x and dx tensors in the layer.IO
//Channel dim is limited to 1-4. If c=1: [r,r,r,255]; If c=2: [r,g,avg(r,g),255]; c=3: [r,g,b,255]; c=4: [r,g,b,a];
func (im *Imager) TileBatchesXdX(handle *cudnn.Handler, x *layers.IO, h, w, hstride, wstride int) (image.Image, image.Image, error) {

	frmt, dtype, dims, err := im.shaper.GetB2SOutputProperties(handle, x.T(), []int32{int32(h), int32(w)}, []int32{int32(hstride), int32(wstride)})
	if err != nil {
		return nil, nil, err
	}
	if im.cache == nil {
		_, _, xdims, err := x.Properties()
		if err != nil {
			return nil, nil, err
		}

		xdims[0] = handle.GetMaxBatch()
		im.cache, err = tensor.BuildWithMaxDims(handle, frmt, dtype, dims, xdims)
		if err != nil {
			return nil, nil, err
		}
	} else {
		_, _, cdims, err := im.cache.Properties()
		if err != nil {
			return nil, nil, err
		}
		if utils.CompareInt32(dims, cdims) == false {

			err = im.cache.ChangeDims(dims)
			if err != nil {
				return nil, nil, err
			}
		}
	}
	err = im.shaper.S2BBackward(handle, im.cache, x.T(), []int32{int32(hstride), int32(wstride)})
	if err != nil {
		return nil, nil, err
	}
	var dflg cudnn.DataTypeFlag
	vol := utils.FindVolumeInt32(dims, nil)
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
		return nil, nil, errors.New("Unsupported Datatype")
	}
	err = im.cache.Memer().FillSlice(z)
	if err != nil {
		return nil, nil, err
	}
	img, err := makeimage(z, dims, frmt)
	if err != nil {
		return nil, nil, err
	}
	err = im.shaper.S2BBackward(handle, im.cache, x.DeltaT(), []int32{int32(hstride), int32(wstride)})
	if err != nil {
		return nil, nil, err
	}
	err = im.cache.Memer().FillSlice(z)
	if err != nil {
		return nil, nil, err
	}
	img2, err := makeimage(z, dims, frmt)
	if err != nil {
		return nil, nil, err
	}
	return img, img2, nil

}

//TileBatches will take the batches and lay place them withing the HWC space like tiles.
//Channel dim is limited to 1-4. If c=1: [r,r,r,255]; If c=2: [r,g,avg(r,g),255]; c=3: [r,g,b,255]; c=4: [r,g,b,a];
func (im *Imager) TileBatches(handle *cudnn.Handler, x *layers.IO, h, w, hstride, wstride int) (image.Image, error) {

	frmt, dtype, dims, err := im.shaper.GetB2SOutputProperties(handle, x.T(), []int32{int32(h), int32(w)}, []int32{int32(hstride), int32(wstride)})
	if err != nil {

		return nil, err
	}
	if im.cache == nil {
		_, _, xdims, err := x.Properties()
		if err != nil {
			return nil, err
		}

		xdims[0] = handle.GetMaxBatch()
		im.cache, err = tensor.BuildWithMaxDims(handle, frmt, dtype, dims, xdims)
		if err != nil {
			return nil, err
		}
	} else {
		_, _, cdims, err := im.cache.Properties()
		if err != nil {
			return nil, err
		}
		if utils.CompareInt32(dims, cdims) == false {

			err = im.cache.ChangeDims(dims)
			if err != nil {
				return nil, err
			}
		}
	}

	err = im.shaper.S2BBackward(handle, im.cache, x.T(), []int32{int32(hstride), int32(wstride)})
	if err != nil {
		return nil, err
	}
	var dflg cudnn.DataTypeFlag
	vol := utils.FindVolumeInt32(dims, nil)
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
	im.cache.Memer().FillSlice(z)
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

func makeimage(data []float32, dims []int32, frmt cudnn.TensorFormat) (image.Image, error) {
	unNormalize(data)
	if len(data) != int(utils.FindVolumeInt32(dims, nil)) {
		return nil, errors.New("Volume Size doesn't Match intput size")
	}
	var fflag cudnn.TensorFormatFlag
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
					g1 := uint8(data[(1*h*w)+(i*w)+j])
					b1 := uint8(data[(2*h*w)+(i*w)+j])
					a := uint8(data[(3*h*w)+(i*w)+j])
					img1.Set(j, i, color.RGBA{r1, g1, b1, a})
				}
			}

		case 3:
			for i := 0; i < h; i++ {
				for j := 0; j < w; j++ {
					r1 := uint8(data[(0*h*w)+(i*w)+j])
					g1 := uint8(data[(1*h*w)+(i*w)+j])
					b1 := uint8(data[(2*h*w)+(i*w)+j])
					img1.Set(j, i, color.RGBA{r1, g1, b1, a})
				}
			}
		case 2:
			for i := 0; i < h; i++ {
				for j := 0; j < w; j++ {
					r1 := uint8(data[(0*h*w)+(i*w)+j])
					g1 := uint8(data[(1*h*w)+(i*w)+j])
					b1 := (r1 + g1) / 2
					img1.Set(j, i, color.RGBA{r1, g1, b1, a})
				}
			}

		case 1:
			for i := 0; i < h; i++ {
				for j := 0; j < w; j++ {
					r1 := uint8(data[(i*w)+j])
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
	return utils.FindMin(data)
}
func maxvalue(data []float32) float32 {
	return utils.FindMax(data)

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
