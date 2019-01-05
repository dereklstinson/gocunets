package cpu

import (
	"errors"
)

//XYPoint interface is used to access the xy coordinate of a point
type XYPoint interface {
	XY() (float32, float32)
}

//LabelAssesment is used to assess the points that have been moved because of a reshape
type LabelAssesment struct {
	locinarray         int     //Location In the []Point Array passed in ShapeToBatchLabelAdjustForward
	batchlocation      int     //Location in the batch dim that this point is residing
	x, y               float32 // new xy location in batch
	numberofsharedwith int     //This is the total count of labels shared in the batch
}

//ShapeToBatchLabelAdjustForward is a way to seperate the labels by batch
func ShapeToBatchLabelAdjustForward(dims, window, stride []int32, pts []XYPoint) []LabelAssesment {
	assess := make([]LabelAssesment, len(pts))
	if len(dims) != 4 {
		return nil
	}
	if dims[0] != int32(1) {
		return nil
	}
	h := window[0]
	w := window[1]
	hs := stride[0]
	ws := stride[1]
	n1 := intceiling(dims[2]-h, hs) + 1
	n2 := intceiling(dims[3]-w, ws) + 1

	oh := int32(0)
	for i := int32(0); i < n1; i++ {
		ow := int32(0)
		for j := int32(0); j < n2; j++ {

			miny := float32(oh)
			minx := float32(ow)
			oh += h
			ow += w
			maxy := float32(oh)
			maxx := float32(ow)
			batch := i*n2 + j
			counter := 0
			for k := 0; k < len(pts); k++ {
				x, y := pts[k].XY()
				if x > minx && x < maxx && y > miny && y < maxy {
					counter++
				}

			}
			if counter > 0 {
				for k := 0; k < len(pts); k++ {
					x, y := pts[k].XY()
					if x > minx && x < maxx && y > miny && y < maxy {
						assess[k].batchlocation = int(batch)
						assess[k].x = x - minx
						assess[k].y = y - miny
						assess[k].locinarray = k
						assess[k].numberofsharedwith = counter
					}

				}
			}

		}

	}
	return assess

}

//ShapeToBatchNCHW4DForward Takes a Volume and Segments it into Batches to the size h,w given. and rounds up by one.  Values not used in new tensor will be zero
func ShapeToBatchNCHW4DForward(values []float32, dims, window, stride []int32) ([]float32, []int32, error) {
	if len(dims) != 4 {
		return nil, nil, errors.New("The Length of dims should equal 4")
	}
	if dims[0] != int32(1) {
		return nil, nil, errors.New("N value needs to be 1")
	}
	h := window[0]
	w := window[1]
	hs := stride[0]
	ws := stride[1]
	n1 := intceiling(dims[2]-h, hs) + 1
	n2 := intceiling(dims[3]-w, ws) + 1
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
			striderw += ws
		}
		striderh += hs
	}
	return v, newdims, nil
}

//ShapeToBatchNCHW4DBackward Takes a Volume and Segments it into Batches to the size h,w given. and rounds up by one.  Values not used in new tensor will be zero
func ShapeToBatchNCHW4DBackward(values []float32, dims []int32, batchedvalues []float32, batchgeddims, stride []int32) error {
	if len(dims) != 4 {
		return errors.New("The Length of dims should equal 4")
	}
	if dims[0] != int32(1) {
		return errors.New("N value needs to be 1")
	}
	n1 := intceiling(dims[2]-batchgeddims[2], stride[0]) + 1
	n2 := intceiling(dims[3]-batchgeddims[3], stride[1]) + 1
	oHH := dims[3]
	oHW := dims[2]

	c := dims[1]
	h := batchgeddims[2]
	w := batchgeddims[3]
	for i := range values {
		values[i] = 0
	}
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
							values[(k*oHW*oHH)+(oh*oHW)+(ow)] += batchedvalues[(i*n2*c*h*w)+(j*c*h*w)+(k*h*w)+(l*h)+m]
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

//ShapeToBatchNHWC4DForward Takes a Volume and Segments it into Batches to the size h,w given. and rounds up by one.  Values not used in new tensor will be zero
func ShapeToBatchNHWC4DForward(values []float32, dims, window, stride []int32) (arrangedvalues []float32, newdims []int32, err error) {
	if len(dims) != 4 {
		return nil, nil, errors.New("The Length of dims should equal 4")
	}
	if dims[0] != int32(1) {
		return nil, nil, errors.New("N value needs to be 1")
	}
	h := window[0]
	w := window[1]
	hs := stride[0]
	ws := stride[1]
	n1 := intceiling(dims[1]-h, hs) + 1
	n2 := intceiling(dims[2]-w, ws) + 1

	oHH := dims[1]
	oHW := dims[2]
	n := n1 * n2
	c := dims[3]
	newdims = []int32{n, h, w, c}
	arrangedvalues = make([]float32, n*h*w*c)
	z := int32(0)

	for i, sh := z, z; i < n1; i, sh = i+1, sh+hs {
		for j, sw := z, z; j < n2; j, sw = j+1, sw+ws {
			for l := z; l < h; l++ {
				oh := sh + l
				for m := z; m < w; m++ {
					ow := sw + m
					if oh < oHH && ow < oHW {

						for k := z; k < c; k++ {
							//oh can be thought of as l+(i*h)
							//ow can be thought of as m+(j*w)
							//		fmt.Println(i, j, l, m, oh, ow, k, values[(oh*oHW*c)+(ow*c)+k])
							arrangedvalues[(i*n2*h*w*c)+(j*h*w*c)+(l*w*c)+(m*c)+k] = values[(oh*oHW*c)+(ow*c)+(k)]

							//

						}

					}

				}

			}

		}

	}
	return arrangedvalues, newdims, nil
}

//ShapeToBatchNHWC4DBackward Takes a Volume and Segments it into Batches to the size h,w given. and rounds up by one.  Values not used in new tensor will be zero
func ShapeToBatchNHWC4DBackward(values []float32, dims []int32, batchedvalues []float32, batcheddims, stride []int32) error {
	if len(dims) != 4 {
		return errors.New("The Length of dims should equal 4")
	}
	if dims[0] != int32(1) {
		return errors.New("N value needs to be 1")
	}
	h := batcheddims[1]
	w := batcheddims[2]
	hs := stride[0]
	ws := stride[1]
	n1 := intceiling(dims[1]-h, hs) + 1
	n2 := intceiling(dims[2]-w, ws) + 1

	oHH := dims[1]
	oHW := dims[2]

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
			striderw += ws
		}
		striderh += hs
	}
	return nil
}

//Volume finds the volume
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
