package jpeg

import (
	"io"
	"io/ioutil"

	"github.com/dereklstinson/gocudnn/gocu"
	"github.com/dereklstinson/gocudnn/nvjpeg"
)

//Decoder decodes jpegs int nvjpeg.Images
type Decoder struct {
	state *nvjpeg.JpegState
	h     *nvjpeg.Handle
	s     gocu.Streamer
}

//CreateDecoder creates a decoder
func CreateDecoder(h *nvjpeg.Handle, s gocu.Streamer) (*Decoder, error) {
	state, err := nvjpeg.CreateJpegState(h)
	if err != nil {
		return nil, err
	}
	return &Decoder{
		state: state,
		h:     h,
		s:     s,
	}, nil
}

//DecodeAIO decodes one jpeg through all the phases in on function
func (d *Decoder) DecodeAIO(r io.Reader, frmt nvjpeg.OutputFormat, allocator gocu.Allocator) (*Image, error) {
	data, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	img, err := createEmptyImage(d.h, frmt, data, allocator)
	if err != nil {
		return nil, err
	}
	err = d.state.Decode(d.h, data, frmt, img.img, d.s)
	if err != nil {
		return nil, err
	}

	return img, nil
}

//Phase1 does the first phase of decoding
func (d *Decoder) Phase1(data []byte, frmt nvjpeg.OutputFormat) error {
	return d.state.DecodePhase1(d.h, data, frmt, d.s)
}

//Phase2 does the second phase of decoding
func (d *Decoder) Phase2() error {
	return d.state.DecodePhase2(d.h, d.s)
}

//Phase3 does the third and last phase of decoding.
//dest needs to be preallocated.
func (d *Decoder) Phase3(dest *Image) error {
	return d.state.DecodePhase3(d.h, dest.img, d.s)
}

//DecodeBatchedAIO returns a batch of images
func (d *Decoder) DecodeBatchedAIO(r []io.Reader, maxCPUthreads int, frmt nvjpeg.OutputFormat, allocator gocu.Allocator) ([]*Image, error) {
	imgs := make([]*Image, len(r))
	data := make([][]byte, len(r))
	var err error
	for i := range r {
		imgs[i], err = CreateDestImage(d.h, frmt, r[i], allocator)
		if err != nil {
			return nil, err
		}
		rdata, err := ioutil.ReadAll(r[i])
		if err != nil {
			return nil, err
		}
		data[i] = rdata
	}
	nvimgs := getnvjpegimagearrays(imgs)
	err = d.state.DecodeBatchedInitialize(d.h, len(r), maxCPUthreads, frmt)
	if err != nil {
		return nil, err
	}
	err = d.state.DecodeBatched(d.h, data, nvimgs, d.s)
	if err != nil {
		return nil, err
	}

	return imgs, nil
}

//BatchPhaseInitialize initialezes a batch of phases. It also initializes the bytes(BatchedPhase1) and dests (BatchedPhase3) used for the phases.
func (d *Decoder) BatchPhaseInitialize(r []io.Reader, maxCputhreads int, frmt nvjpeg.OutputFormat, allocator gocu.Allocator) (data [][]byte, dests []*Image, err error) {
	batchsize := len(r)
	dests = make([]*Image, len(r))
	data = make([][]byte, len(r))

	for i := range r {
		dests[i], err = CreateDestImage(d.h, frmt, r[i], allocator)
		if err != nil {
			return nil, nil, err
		}
		rdata, err := ioutil.ReadAll(r[i])
		if err != nil {
			return nil, nil, err
		}
		data[i] = rdata
	}
	return data, dests, d.state.DecodeBatchedInitialize(d.h, batchsize, maxCputhreads, frmt)
}

//BatchedPhase1 does phase one of batch decoding
func (d *Decoder) BatchedPhase1(data []byte, imgidx, threadidx int) error {
	return d.state.DecodeBatchedPhase1(d.h, data, imgidx, threadidx, d.s)
}

//BatchedPhse2 does phase two of batched decoding
func (d *Decoder) BatchedPhse2() error {
	return d.state.DecodePhase2(d.h, d.s)
}

//BatchedPhase3 does phase three of batched decoding
func (d *Decoder) BatchedPhase3(dest []*Image) error {
	nvimgs := getnvjpegimagearrays(dest)
	return d.state.DecodeBatchedPhase3(d.h, nvimgs, d.s)
}
