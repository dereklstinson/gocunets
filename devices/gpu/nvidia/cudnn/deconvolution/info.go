package deconvolution

import gocudnn "github.com/dereklstinson/gocudnn"

//Info is the contains the info to make the op
type Info struct {
	CMode       gocudnn.ConvolutionMode `json:"c_mode,omitempty"`
	Dtype       gocudnn.DataType        `json:"dtype,omitempty"`
	Pad         []int32                 `json:"pad,omitempty"`
	Stride      []int32                 `json:"stride,omitempty"`
	Dilation    []int32                 `json:"dilation,omitempty"`
	Format      gocudnn.TensorFormat    `json:"format,omitempty"`
	MathType    gocudnn.MathType        `json:"math_type,omitempty"`
	FwdAlgo     ForwardPerformance      `json:"fwd_algo,omitempty"`
	BwdDataAlgo BackDataPerformance     `json:"bwd_data_algo,omitempty"`
	BwdFiltAlgo BackFilterPerformance   `json:"bwd_filt_algo,omitempty"`
	Group       int32                   `json:"group,omitempty"`
}

//Stage stages/sets up an Ops and returns a pointer to it with the info stored in the info type
func (input Info) Stage() (*Ops, error) {
	op, err := gocudnn.CreateDeConvolutionDescriptor()
	if err != nil {
		return nil, err
	}
	err = op.Set(input.CMode, input.Dtype, input.Pad, input.Stride, input.Dilation)
	if err != nil {
		return nil, err
	}
	err = op.SetMathType(input.FwdAlgo.MathType)
	if err != nil {
		return nil, err
	}
	err = op.SetGroupCount(input.Group)
	if err != nil {
		return nil, err
	}
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
