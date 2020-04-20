package neuralhelpers

import (
	"errors"
	"github.com/anthonynsimon/bild/transform"
	//	gocunets "github.com/dereklstinson/GoCuNets"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/cutil"
	"sync"

	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/cvdh"
	"image"
	"image/jpeg"
	"io"
	"math/rand"
)

//GetImages gets images
func GetImages(r []io.Reader) []image.Image {
	imgs := make([]image.Image, len(r))
	var err error
	for i := range r {
		imgs[i], err = jpeg.Decode(r[i])
		if err != nil {
			panic(err)
		}
	}
	return imgs
}

//GPULoader loads the gpu
type GPULoader struct {
	buffer []*nvidia.Malloced
	i, j   int
	sib    uint
	w      *gocu.Worker
	s      *cudart.Stream
	dev    cudart.Device
	mck    cudart.MemcpyKind
	mux    *sync.Mutex
	err    chan error
}

//CreateGpuLoader creates a gpu loader
func CreateGpuLoader(dev cudart.Device, nbuffer int, SIBperbuff uint) (g *GPULoader, err error) {
	g = new(GPULoader)
	g.mck.Default()
	g.buffer = make([]*nvidia.Malloced, nbuffer)
	g.sib = SIBperbuff
	g.dev = dev
	g.mux = new(sync.Mutex)
	g.w = gocu.NewWorker(dev)
	g.s, err = cudart.CreateNonBlockingStream()
	if err != nil {
		return nil, errors.New("CreateGpuLoader in making stream")
	}

	for i := range g.buffer {
		err = cudart.MallocManagedGlobal(g.buffer[i], SIBperbuff)
		if err != nil {
			return nil, err
		}
	}

	return g, nil
}
func (g *GPULoader) SyncStream() error {
	return g.s.Sync()
}
func (g *GPULoader) AsyncLoadMem(dest cutil.Pointer) {

	go func() {
		err := g.w.Work(func() error {
			g.mux.Lock()
			defer g.mux.Unlock()
			if g.i%len(g.buffer) != g.j%len(g.buffer) {
				err := cudart.MemcpyAsync(dest, g.buffer[g.i], g.sib, g.mck, g.s)
				if err != nil {
					return err
				}
				g.i++
				return nil

			}
			return errors.New("Buffer locked up")

		})
		if err != nil {
			panic(err)
		}
	}()

}
func (g *GPULoader) LoadBuffer(tensor *cvdh.Tensor4d) {
	go func() {
		err := g.w.Work(func() error {
			g.mux.Lock()
			defer g.mux.Unlock()
			if g.i%len(g.buffer) != g.j+1%len(g.buffer) {
				g.j++
				offset := g.j % len(g.buffer)
				gmem, err := gocu.MakeGoMem(tensor.Data)
				if err != nil {
					return err
				}
				err = cudart.MemcpyAsync(g.buffer[offset], gmem, g.sib, g.mck, g.s)
				if err != nil {
					return err
				}
				return nil
			}
			panic("LoadBuffer Locked")
		})
		if err != nil {
			panic(err)
		}
	}()
}

//CreateRandCropBatch creates a RandCropBatch
func CreateRandCropBatch(input, output []image.Image, h, w, resizedouth, resizedoutw int) (tin, tout *cvdh.Tensor4d) {
	var wg sync.WaitGroup
	for i := range input {
		wg.Add(1)
		go func(i int) {
			y := input[i].Bounds().Max.Y
			x := input[i].Bounds().Max.X
			difx := x - w
			dify := y - h
			if difx < 0 || dify < 0 {
				panic(" difx<0||dify<0")
			}

			x0 := rand.Int() % (difx + 1)
			y0 := rand.Int() % (dify + 1)
			x1 := x0 + w
			y1 := y0 + h
			input[i] = transform.Crop(input[i], image.Rect(x0, y0, x1, y1))
			output[i] = transform.Crop(output[i], image.Rect(x0, y0, x1, y1))
			output[i] = transform.Resize(output[i], resizedoutw, resizedouth, transform.NearestNeighbor)
			wg.Done()
		}(i)

	}
	wg.Wait()
	tin = cvdh.Create4dTensor(input, true)
	tout = cvdh.Create4dTensor(output, true)
	return tin, tout
}
