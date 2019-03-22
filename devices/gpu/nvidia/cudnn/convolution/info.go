package convolution

import gocudnn "github.com/dereklstinson/GoCudnn"

//Info is the contains the info to make the op
type Info struct {
	CMode       gocudnn.ConvolutionMode `json:"c_mode,omitempty"`
	Dtype       gocudnn.DataType        `json:"dtype,omitempty"`
	Pad         []int32                 `json:"pad,omitempty"`
	Stride      []int32                 `json:"stride,omitempty"`
	Dilation    []int32                 `json:"dilation,omitempty"`
	FwdAlgo     ForwardPerformance      `json:"fwd_algo,omitempty"`
	BwdDataAlgo BackDataPerformance     `json:"bwd_data_algo,omitempty"`
	BwdFiltAlgo BackFilterPerformance   `json:"bwd_filt_algo,omitempty"`
	Group       int32                   `json:"group,omitempty"`
}

//Stage stages/sets up an Ops and returns a pointer to it with the info stored in the info type
func (input Info) Stage() (*Ops, error) {
	op, err := gocudnn.CreateConvolutionDescriptor()
	if err != nil {
		return nil, err
	}
	err = op.Set(input.CMode, input.Dtype, input.Pad, input.Stride, input.Dilation)
	if err != nil {
		return nil, err
	}
	op.SetMathType(input.FwdAlgo.MathType)

	return &Ops{
		op:           op,
		perfforward:  input.FwdAlgo,
		perfbackdata: input.BwdDataAlgo,
		perfbackfilt: input.BwdFiltAlgo,
		group:        input.Group,
		stride:       input.Stride,
		dilation:     input.Dilation,
		pad:          input.Pad,
	}, nil
}

//Info returns an info struct and error.  Info is usually used for saving the data to a json file.
func (c *Ops) Info() (Info, error) {
	mode, dtype, pad, stride, dilation, err := c.op.Get()
	return Info{
		CMode:       mode,
		Dtype:       dtype,
		Pad:         pad,
		Stride:      stride,
		Dilation:    dilation,
		FwdAlgo:     c.perfforward,
		BwdDataAlgo: c.perfbackdata,
		BwdFiltAlgo: c.perfbackfilt,
	}, err
}
