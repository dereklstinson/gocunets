package tensor

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

/*

//ToOneImageColor will return an image.Image of the volume in batch/neuron for the rows, and channels for the column
//Y and X represent how much padding of (hopefully black) there is between the channels and neurons
func (t *Volume) ToOneImageColor(X, Y int) (image.Image, error) {
	images, err := t.ToImagesColor()
	if err != nil {
		return nil, err
	}
	return ToOneImage(images, X, Y), nil
}

//ToImagesColor makes a 2d array of images. Negative will be Green. Positive will be purple it returns a double array of image.Image. Only Float32 support right now
func (t *Volume) ToImagesColor() ([][]image.Image, error) {
	return t.convertColor()
}


*/
