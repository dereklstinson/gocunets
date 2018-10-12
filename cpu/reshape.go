package cpu

import (
	"errors"
)

//ShapeToBatchNCHW4DForward Takes a Volume and Segments it into Batches to the size h,w given. and rounds up by one.  Values not used in new tensor will be zero
func ShapeToBatchNCHW4DForward(values []float32, dims []int32, h, w int32) ([]float32, []int32, error) {
	if len(dims) != 4 {
		return nil, nil, errors.New("The Length of dims should equal 4")
	}
	if dims[0] != int32(1) {
		return nil, nil, errors.New("N value needs to be 1")
	}
	n1 := intceiling(dims[2], h)
	n2 := intceiling(dims[3], w)
	oHH := dims[2]
	oHW := dims[3]
	n := n1 * n2
	c := dims[1]
	newdims := []int32{n, c, h, w}
	//	totalvol := Volume(dims)
	v := make([]float32, n*c*h*w)
	striderh := int32(0)
	for i := int32(0); i < n1; i++ {
		striderw := int32(0)
		for j := int32(0); j < n2; j++ {
			for k := int32(0); k < c; k++ {
				for l := int32(0); l < h; l++ {
					oh := striderh + l
					for m := int32(0); m < w; m++ {
						ow := striderw + m
						if oh < oHH && ow < oHW {
							v[(i*n2*c*h*w)+(j*c*h*w)+(k*h*w)+(l*h)+m] = values[(k*oHW*oHH)+(oh*oHW)+(ow)]
						} else {
							v[(i*n2*c*h*w)+(j*c*h*w)+(k*h*w)+(l*h)+m] = 0
						}

					}
				}
			}
			striderw += w
		}
		striderh += h
	}
	return v, newdims, nil
}

//func BatchToShapeNCHW4D(batchedvalues []float32, batchgeddims []int32, h, w int32)([]float32, []int32, error)
//ShapeToBatchNCHW4DBackward Takes a Volume and Segments it into Batches to the size h,w given. and rounds up by one.  Values not used in new tensor will be zero
func ShapeToBatchNCHW4DBackward(values []float32, dims []int32, batchedvalues []float32, batchgeddims []int32) error {
	if len(dims) != 4 {
		return errors.New("The Length of dims should equal 4")
	}
	if dims[0] != int32(1) {
		return errors.New("N value needs to be 1")
	}
	n1 := intceiling(dims[2], batchgeddims[2])
	n2 := intceiling(dims[3], batchgeddims[3])
	oHH := dims[3]
	oHW := dims[2]

	c := dims[1]
	h := batchgeddims[2]
	w := batchgeddims[3]
	//testarray := make([]float32, oHH*oHW*c)
	striderh := int32(0)
	for i := int32(0); i < n1; i++ {
		striderw := int32(0)
		for j := int32(0); j < n2; j++ {
			for k := int32(0); k < c; k++ {
				for l := int32(0); l < h; l++ {
					oh := striderh + l
					for m := int32(0); m < w; m++ {
						ow := striderw + m
						if oh < oHH && ow < oHW {
							//testarray[(k*oHW*oHH)+(oh*oHW)+(ow)] = batchedvalues[(i*n2*c*h*w)+(j*c*h*w)+(k*h*w)+(l*h)+m]
							values[(k*oHW*oHH)+(oh*oHW)+(ow)] = batchedvalues[(i*n2*c*h*w)+(j*c*h*w)+(k*h*w)+(l*h)+m]
						}

					}
				}
			}
			striderw += w
		}
		striderh += h
	}

	return nil
}

/*
//ShapeToBatchNCHW4DBackward Takes a Volume and Segments it into Batches to the size h,w given. and rounds up by one.  Values not used in new tensor will be zero
func ShapeToBatchNCHW4DBackward(values []float32, dims []int32, h, w int32) ([]float32, []int32, error) {
	if len(dims) != 4 {
		return nil, nil, errors.New("The Length of dims should equal 4")
	}
	if dims[0] != int32(1) {
		return nil, nil, errors.New("N value needs to be 1")
	}
	n1 := intceiling(h, dims[2])
	n2 := intceiling(w, dims[3])
	oHH := h
	oHW := w

	c := dims[1]
	h = dims[2]
	w = dims[3]
	newdims := []int32{1, c, h, w}
	v := make([]float32, 1*c*h*w)

	striderh := int32(0)
	for i := int32(0); i < n1; i++ {
		striderw := int32(0)
		for j := int32(0); j < n2; j++ {
			for k := int32(0); k < c; k++ {
				for l := int32(0); l < h; l++ {
					oh := striderh + l
					for m := int32(0); m < w; m++ {
						ow := striderw + m
						if oh < oHH && ow < oHW {
							v[(k*oHW*oHH)+(oh*oHW)+(ow)] = values[(i*n2*c*h*w)+(j*c*h*w)+(k*h*w)+(l*h)+m]
						}

					}
				}
			}
			striderw += w
		}
		striderh += h
	}
	return v, newdims, nil
}
*/
//ShapeToBatchNHWC4DForward Takes a Volume and Segments it into Batches to the size h,w given. and rounds up by one.  Values not used in new tensor will be zero
func ShapeToBatchNHWC4DForward(values []float32, dims []int32, h, w int32) ([]float32, []int32, error) {
	if len(dims) != 4 {
		return nil, nil, errors.New("The Length of dims should equal 4")
	}
	if dims[0] != int32(1) {
		return nil, nil, errors.New("N value needs to be 1")
	}

	n1 := intceiling(dims[1], h)
	n2 := intceiling(dims[2], w)
	oHH := dims[1]
	oHW := dims[2]
	n := n1 * n2
	c := dims[3]
	newdims := []int32{n, h, w, c}
	v := make([]float32, n*h*w*c)
	z := int32(0)

	for i, sh := z, z; i < n1; i, sh = i+1, sh+h {
		for j, sw := z, z; j < n2; j, sw = j+1, sw+w {
			for l := z; l < h; l++ {
				oh := sh + l
				for m := z; m < w; m++ {
					ow := sw + m
					if oh < oHH && ow < oHW {

						for k := z; k < c; k++ {
							//		fmt.Println(i, j, l, m, oh, ow, k, values[(oh*oHW*c)+(ow*c)+k])
							v[(i*n2*h*w*c)+(j*h*w*c)+(l*w*c)+(m*c)+k] = values[(oh*oHW*c)+(ow*c)+(k)]
						}

					}

				}

			}

		}

	}
	return v, newdims, nil
}

//ShapeToBatchNHWC4DBackward Takes a Volume and Segments it into Batches to the size h,w given. and rounds up by one.  Values not used in new tensor will be zero
func ShapeToBatchNHWC4DBackward(values []float32, dims []int32, batchedvalues []float32, batcheddims []int32) error {
	if len(dims) != 4 {
		return errors.New("The Length of dims should equal 4")
	}
	if dims[0] != int32(1) {
		return errors.New("N value needs to be 1")
	}
	n1 := intceiling(dims[1], batcheddims[1])
	n2 := intceiling(dims[2], batcheddims[2])
	oHH := dims[1]
	oHW := dims[2]
	h := batcheddims[1]
	w := batcheddims[2]
	c := dims[3]

	striderh := int32(0)
	for i := int32(0); i < n1; i++ {
		striderw := int32(0)
		for j := int32(0); j < n2; j++ {

			for l := int32(0); l < h; l++ {
				oh := striderh + l
				for m := int32(0); m < w; m++ {
					ow := striderw + m
					if oh < oHH && ow < oHW {
						for k := int32(0); k < c; k++ {
							values[(oh*oHW*c)+(ow*c)+(k)] = batchedvalues[(i*n2*c*h*w)+(j*c*h*w)+(l*w*c)+(m*c)+k]
						}
					}
				}
			}
			striderw += w
		}
		striderh += h
	}
	return nil
}

func Volume(dims []int32) int32 {
	vol := int32(1)
	for i := 0; i < len(dims); i++ {
		vol *= dims[i]
	}
	return vol
}
func intceiling(a, b int32) int32 {
	return ((a - int32(1)) / b) + int32(1)
}
