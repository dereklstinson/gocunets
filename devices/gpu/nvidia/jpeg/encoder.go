package jpeg

import (
	"github.com/dereklstinson/gocudnn/gocu"
	"github.com/dereklstinson/gocudnn/nvjpeg"
)

//MakeHandle creates an jpeg handle
//if devbuff == 0 then default setting will be used
//if pinnedbuff == 0 then default setting will be used
func MakeHandle(devbuff, pinnedbuff uint) (*nvjpeg.Handle, error) {
	var beflag nvjpeg.Backend
	h, err := nvjpeg.CreateEx(beflag.Default())
	if err != nil {

		return nil, err
	}
	if devbuff != 0 {
		err = h.SetDeviceMemoryPadding(devbuff)
		if err != nil {

			return nil, err
		}
	}
	if pinnedbuff != 0 {
		err = h.SetPinnedMemoryPadding(pinnedbuff)
		if err != nil {

			return nil, err
		}
	}

	return h, nil
}

//Encoder streamlines nvjpegs encoderparams, encoderstate, handle, and gocu.Streamer.
type Encoder struct {
	params *nvjpeg.EncoderParams
	state  *nvjpeg.EncoderState
	s      gocu.Streamer
	h      *nvjpeg.Handle
}

//CreateEncoder creates an encoder
func CreateEncoder(h *nvjpeg.Handle, s gocu.Streamer) (*Encoder, error) {
	params, err := nvjpeg.CreateEncoderParams(h, s)
	if err != nil {
		return nil, err
	}
	state, err := nvjpeg.CreateEncoderState(h, s)
	if err != nil {
		return nil, err
	}
	return &Encoder{
		params: params,
		state:  state,
		s:      s,
		h:      h,
	}, nil
}

//GetBufferSize - Returns the maximum possible buffer size that is needed to store the
//compressed JPEG stream, for the given input parameters.
func (e *Encoder) GetBufferSize(width, height int32) (maxStreamLength uint, err error) {
	return e.params.GetBufferSize(e.h, width, height)
}

//SetSamplingFactors -Sets which chroma subsampling will be used for JPEG compression.
//ssfactor that will be used for JPEG compression.
//If the input is in YUV color model and ssfactor is different from the subsampling factors
//of source image, then the NVJPEG library will convert subsampling to the value of chroma_subsampling.
// Default value is 4:4:4.
func (e *Encoder) SetSamplingFactors(factor nvjpeg.ChromaSubsampling) error {
	return e.params.SetSamplingFactors(factor, e.s)
}

//SetOptimizedHuffman - Sets whether or not to use optimized Huffman.
//Using optimized Huffman produces smaller JPEG bitstream sizes with
//the same quality, but with slower performance.
//Default is false
func (e *Encoder) SetOptimizedHuffman(optimized bool) error {
	return e.params.SetOptimizedHuffman(optimized, e.s)
}

//SetQuality sets the quality of the paramters of the endcoder
//Quality should be a number between 1 and 100. The default is set to 70
func (e *Encoder) SetQuality(quality int32) error {
	return e.params.SetQuality(quality, e.s)
}

//EncodeYUV -Compresses the image in YUV colorspace to JPEG stream using the paramters set by encoder,
//and stores it in the Encoder.
func (e *Encoder) EncodeYUV(src *nvjpeg.Image, srcChroma nvjpeg.ChromaSubsampling, swidth, sheight int32) error {
	return e.state.EncodeYUV(e.h, e.params, src, srcChroma, swidth, sheight, e.s)
}

//EncodeImage - Compresses the image in the provided format to the JPEG stream using the paramters set by encoder,
//and stores it in the Encoder.
func (e *Encoder) EncodeImage(src *nvjpeg.Image, fmt nvjpeg.InputFormat, swidth, sheight int32) error {
	return e.state.EncodeImage(e.h, e.params, src, fmt, swidth, sheight, e.s)
}

//Read reads up to len(p). n is the number of bytes read.  if len(p) < buffer stream function will not read any bytes.
//error sould return number of that should be made for p, or it will contain bigger error issues from nvjpeg.
//Use GetByteSize() to get the size of p.
func (e *Encoder) Read(p []byte) (n int, err error) {
	return e.state.ReadBitStream(e.h, p, e.s)
}

//GetByteSize returns the number of bytes in compressed buffer. Can be used to make p for Read()
func (e *Encoder) GetByteSize() (uint, error) {
	return e.state.GetCompressedBufferSize(e.h, e.s)
}
