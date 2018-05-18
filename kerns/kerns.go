package kerns

import (
	"fmt"
	"time"

	"github.com/dereklstinson/cuda"
)

func Testfunc() error {
	var ctx *cuda.Context

	devices, err := cuda.AllDevices()
	_ = MakeMakeFile("/home/derek/go/src/github.com/dereklstinson/kernels/cu/", "kernels32.cu", *devices[0])
	fmt.Println(devices[0].Name())
	fmt.Println(devices[0].TotalMem())
	/*
		PrintAllDeviceAttributes(*devices[0])
		if err != nil {
			panic(err)
		}
	*/
	if len(devices) == 0 {
		panic("Number of devices is 0")
	}
	inputx := uint(32)
	inputy := uint(32)
	inputz := uint(3)
	inputd := uint(80)
	kernelx := uint(5)
	kernely := uint(5)
	kernelz := uint(3)
	kerneld := inputd
	neurons := uint(80)
	slidex := uint(1)
	slidey := uint(1)
	slidez := uint(1)
	paddingx := uint(1)
	paddingy := uint(1)
	paddingz := uint(0)
	outputx := ((inputx - kernelx + (2 * paddingx)) / slidex) + 1
	outputy := ((inputy - kernely + (2 * paddingy)) / slidey) + 1
	outputz := ((inputz - kernelz + (2 * paddingz)) / slidez) + 1
	matrix1 := SampleMatrix4d(inputx, inputy, inputz, inputd, 2.0)

	matrix2 := SampleNeuronLayer4d(neurons, kernelx, kernely, kernelz, kerneld, 1.0)

	matrix3 := SampleMatrix4d(neurons, outputx, outputy, outputz, 0.0)
	matrix4 := SampleMatrix4d(neurons, outputx, outputy, outputz, 0.0)
	biases := SampleMatrix4d(neurons, 1, 1, 1, 1.0)

	ctx, err = cuda.NewContext(devices[0], -1)
	if err != nil {
		panic(err) // Handle error.
	}
	buffer, err := cuda.LoadToDevice(matrix1)
	if err != nil {
		return err
	}
	buffer1, err := cuda.LoadToDevice(matrix2)
	if err != nil {
		return err
	}
	buffer2, err := cuda.LoadToDevice(matrix3)
	if err != nil {
		return err
	}
	buffer3, err := cuda.LoadToDevice(matrix3)
	if err != nil {
		return err
	}
	bufbias, err := cuda.LoadToDevice(biases)
	if err != nil {
		return err
	}
	err = <-ctx.Run(func() error {

		fmt.Println(cuda.MemInfo())
		devicestream, err := cuda.NewStream(false)
		if err != nil {
			return err
		}
		defer devicestream.Close()

		samplemod, err := cuda.NewModule(ctx, LoadPTXFile("/home/derek/go/src/github.com/dereklstinson/kernels/cu/kernels32.ptx"))
		if err != nil {
			fmt.Println("err in new module")
			return err
		}

		fmt.Println(outputx, outputy, outputz, kernelx, kernely, kernelz, inputx, inputy, inputz)
		sharedmemsize := (2 * kernelx * kernely * kernelz * inputd) * 4
		dim3grid := cuda.SetDims(outputx, outputy, outputz)
		dim3block := cuda.SetDims(kernelx, kernely, kernelz)
		atimer := start()
		zero := uint(0)
		for i := zero; i < neurons; i++ {
			err = samplemod.LaunchDims("convolution4dneuron", dim3grid, dim3block, sharedmemsize, devicestream, buffer.Ptr, buffer1.Ptr, bufbias.Ptr, buffer3.Ptr, inputx, inputy, inputz, paddingx, paddingy, paddingz, slidex, slidey, slidez, inputd, int(i))
			if err != nil {
				fmt.Println("err in this kernel for loop", err)
				return err
			}
		}

		err = buffer3.CudaMemCopyDeviceToHost(matrix4)
		if err != nil {
			fmt.Println("error in reading buffer2", err)
		}

		sectionedneurons := atimer.stop()

		atimer.start()
		err = samplemod.LaunchDims("convolution4dlayer", dim3grid, dim3block, sharedmemsize, devicestream, buffer.Ptr, buffer1.Ptr, bufbias.Ptr, buffer2.Ptr, inputx, inputy, inputz, paddingx, paddingy, paddingz, slidex, slidey, slidez, inputd, neurons)
		if err != nil {
			fmt.Println("error in reading buffer2", err)
		}
		err = buffer2.CudaMemCopyDeviceToHost(matrix3)
		if err != nil {
			fmt.Println("error in reading buffer2", err)
		}

		fullayer := atimer.stop()

		fmt.Println(fullayer, sectionedneurons)
		fmt.Println(cuda.MemInfo())
		return nil
	})

	if err != nil {
		panic(err)
	}
	buffer1.CudaFree()
	fmt.Println(cuda.MemInfo())
	buffer.CudaFree()
	fmt.Println(cuda.MemInfo())
	buffer2.CudaFree()
	fmt.Println(cuda.MemInfo())
	bufbias.CudaFree()
	fmt.Println(cuda.MemInfo())
	buffer3.CudaFree()
	fmt.Println(cuda.MemInfo())

	return nil

}

type timer struct {
	starttime int64
}

func start() timer {
	var t timer
	t.starttime = time.Now().UnixNano()
	return t
}

func (t *timer) stop() int64 {
	return time.Now().UnixNano() - t.starttime
}
func (t *timer) start() {
	t.starttime = time.Now().UnixNano()
}
func compairmatrix(mat1, mat2 []float32) {
	for i := 0; i < len(mat1); i++ {
		if mat1[i] != mat2[i] {
			fmt.Println("Doesn't Match")
			return
		}
	}
}
