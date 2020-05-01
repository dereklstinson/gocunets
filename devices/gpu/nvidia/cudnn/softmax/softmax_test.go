/*
Package softmax uses the softmax functions from gocudnn which is from cudnn.   Except it doesn't use any of the flags.
*/
package softmax

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"

	gocudnn "github.com/dereklstinson/gocudnn"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/tensor"
	"github.com/dereklstinson/gocudnn/cudart"
	"github.com/dereklstinson/gocudnn/gocu"
)

func TestSoftMax(t *testing.T) {
	ndevs, err := cudart.GetDeviceCount()
	if err != nil {
		t.Error(err)
	}
	devs := make([]cudart.Device, ndevs)

	streams := make([]*cudart.Stream, len(devs))
	handles := make([]*cudnn.Handler, len(devs))
	smds := make([]*Ops, len(streams))
	rand.Seed(time.Now().UnixNano())
	var vals []float32
	var dims []int32
	var frmt gocudnn.TensorFormat
	var dtype gocudnn.DataType
	frmt.NCHW()
	dtype.Float()

	fflg := frmt
	switch frmt {
	case fflg.NCHW():
		dh := int32(2)
		dw := int32(2)
		nhws := int32(4)
		dims = []int32{1, nhws, dh, dw}
		hw := make([][]float32, dh*dw)
		hw[0], hw[1], hw[2], hw[3] = []float32{1, 0, 0, 0}, []float32{0, 1, 0, 0}, []float32{0, 0, 1, 0}, []float32{0, 0, 0, 1}
		for i := range hw {
			vals = append(vals, hw[i]...)
		}
	case fflg.NHWC():
		nchan := int32(4)
		dims = []int32{1, 3, 3, nchan}
		channels := make([][]float32, nchan)
		channels[0], channels[1], channels[2], channels[3] = []float32{3, -1, -1, -1}, []float32{-1, 3, -1, -1}, []float32{-1, -1, 3, -1}, []float32{-1, -1, -1, 3}
		for i := range channels {
			vals = append(vals, channels[i]...)
		}
	}
	valptr, err := gocu.MakeGoMem(vals)
	if err != nil {
		t.Error(err)
	}
	for i := range devs {
		devs[i], err = cudart.GetDevice()
		if err != nil {
			t.Error(err)
		}

		handles[i] = cudnn.CreateHandler(gocu.NewWorker(devs[i]), devs[i], rand.Uint64())
		streams[i], err = cudart.CreateNonBlockingStream()
		if err != nil {
			t.Error(err)
		}
		if i%2 == 1 {
			smds[i] = StageLogPerChannel()
		} else {
			smds[i] = StageFastPerChannel()
		}

	}
	xs, ys := make([]*tensor.Volume, ndevs), make([]*tensor.Volume, ndevs)
	for i := range xs {
		xs[i], err = tensor.Build(handles[i], frmt, dtype, dims)
		if err != nil {
			t.Error(err)
		}
		xs[i].LoadMem(handles[i], valptr, valptr.TotalBytes())
		ys[i], err = tensor.Build(handles[i], frmt, dtype, dims)
		if err != nil {
			t.Error(err)
		}
	}
	var wg sync.WaitGroup
	for i, sm := range smds {
		wg.Add(1)
		go func(i int, sm *Ops) {
			err := sm.ForwardProp(handles[i], 1, xs[i], 0, ys[i])
			if err != nil {
				t.Error(err)
			}
			err = handles[i].Sync()
			if err != nil {
				t.Error(err)
			}
			wg.Done()
		}(i, sm)

	}
	wg.Wait()
	yslices := make([][]float32, len(handles))
	for i, y := range ys {
		yslices[i] = make([]float32, y.Vol())
		y.TogglePrintValueForStringer()
		fmt.Println("y", i)
		fmt.Println(y)

		y.FillSlice(handles[i], yslices[i])
		fmt.Println(yslices[i])
		y.TogglePrintValueForStringer()
	}
	var adder float32
	for _, yslice := range yslices {
		for _, value := range yslice {
			adder = adder + value
		}
	}
	if adder > 4.0 {
		t.Error("adder should be less than 4.0")
	}

}
