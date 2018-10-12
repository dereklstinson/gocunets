package cpu

import "errors"

//ShapeToBatchNCHW4DForward Takes a Volume and Segments it into Batches to the size h,w given. and rounds up by one.  Values not used in new tensor will be zero
func ShapeToBatchNCHW4DForward(values []float32, dims []int32, h, w int32) ([]float32, []int32, error) {
	if len(dims) != 4 {
		return nil, nil, errors.New("The Length of dims should equal 4")
	}
	if dims[0] != int32(1) {
		return nil, nil, errors.New("N value needs to be 1")
	}
	n1 := intceiling(dims[2], h)
	n2 := intceiling(dims[3], h)
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

//ShapeToBatchNCHW4DBackward Takes a Volume and Segments it into Batches to the size h,w given. and rounds up by one.  Values not used in new tensor will be zero
func ShapeToBatchNCHW4DBackward(values []float32, dims []int32, h, w int32) ([]float32, []int32, error) {
	if len(dims) != 4 {
		return nil, nil, errors.New("The Length of dims should equal 4")
	}
	if dims[0] != int32(1) {
		return nil, nil, errors.New("N value needs to be 1")
	}
	n1 := intceiling(dims[2], h)
	n2 := intceiling(dims[3], h)
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
							values[(k*oHW*oHH)+(oh*oHW)+(ow)] = v[(i*n2*c*h*w)+(j*c*h*w)+(k*h*w)+(l*h)+m]
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

//ShapeToBatchNHWC4DForward Takes a Volume and Segments it into Batches to the size h,w given. and rounds up by one.  Values not used in new tensor will be zero
func ShapeToBatchNHWC4DForward(values []float32, dims []int32, h, w int32) ([]float32, []int32, error) {
	if len(dims) != 4 {
		return nil, nil, errors.New("The Length of dims should equal 4")
	}
	if dims[0] != int32(1) {
		return nil, nil, errors.New("N value needs to be 1")
	}
	n1 := intceiling(dims[2], h)
	n2 := intceiling(dims[3], h)
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

			for l := int32(0); l < h; l++ {
				oh := striderh + l
				for m := int32(0); m < w; m++ {
					ow := striderw + m
					if oh < oHH && ow < oHW {
						for k := int32(0); k < c; k++ {
							v[(i*n2*c*h*w)+(j*c*h*w)+(l*h*c)+(l*c)+k] = values[(oh*oHW*c)+(ow*c)+(k)]
						}
					} else {
						for k := int32(0); k < c; k++ {
							v[(i*n2*c*h*w)+(j*c*h*w)+(l*h*c)+(l*c)+k] = 0
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

//ShapeToBatchNHWC4DBackward Takes a Volume and Segments it into Batches to the size h,w given. and rounds up by one.  Values not used in new tensor will be zero
func ShapeToBatchNHWC4DBackward(values []float32, dims []int32, h, w int32) ([]float32, []int32, error) {
	if len(dims) != 4 {
		return nil, nil, errors.New("The Length of dims should equal 4")
	}
	if dims[0] != int32(1) {
		return nil, nil, errors.New("N value needs to be 1")
	}
	n1 := intceiling(dims[2], h)
	n2 := intceiling(dims[3], h)
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

			for l := int32(0); l < h; l++ {
				oh := striderh + l
				for m := int32(0); m < w; m++ {
					ow := striderw + m
					if oh < oHH && ow < oHW {
						for k := int32(0); k < c; k++ {
							values[(oh*oHW*c)+(ow*c)+(k)] = v[(i*n2*c*h*w)+(j*c*h*w)+(l*h*c)+(l*c)+k]
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
