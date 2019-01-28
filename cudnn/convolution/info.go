package convolution

import gocudnn "github.com/dereklstinson/GoCudnn"

//Info is the contains the info to make the op
type Info struct {
	CMode        gocudnn.ConvolutionMode `json:"c_mode,omitempty"`
	Dtype        gocudnn.DataType        `json:"dtype,omitempty"`
	Pad          []int32                 `json:"pad,omitempty"`
	Stride       []int32                 `json:"stride,omitempty"`
	Dilation     []int32                 `json:"dilation,omitempty"`
	FwdAlgo      ForwardPerformance      `json:"fwd_algo,omitempty"`
	FwdGroup     int32                   `json:"fwd_group,omitempty"`
	BwdDataAlgo  BackDataPerformance     `json:"bwd_data_algo,omitempty"`
	BwdDataGroup int32                   `json:"bwd_data_group,omitempty"`
	BwdFiltAlgo  BackFilterPerformance   `json:"bwd_filt_algo,omitempty"`
	BwdFiltGroup int32                   `json:"bwd_filt_group,omitempty"`
}

//Stage stages/sets up an Ops and returns a pointer to it with the info stored in the info type
func (input Info) Stage() (*Ops, error) {
	helper := gocudnn.Convolution{}
	if len(input.Pad) == 2 {
		fwddesc, err := helper.NewConvolution2dDescriptor(input.CMode, input.Dtype, input.Pad, input.Stride, input.Dilation)
		if err != nil {
			return nil, err
		}
		bwdddesc, err := helper.NewConvolution2dDescriptor(input.CMode, input.Dtype, input.Pad, input.Stride, input.Dilation)
		if err != nil {
			return nil, err
		}

		bwdfdesc, err := helper.NewConvolution2dDescriptor(input.CMode, input.Dtype, input.Pad, input.Stride, input.Dilation)
		if err != nil {
			return nil, err
		}
		fwddesc.SetMathType(input.FwdAlgo.MathType)
		bwdfdesc.SetMathType(input.BwdFiltAlgo.MathType)
		bwdddesc.SetMathType(input.BwdDataAlgo.MathType)
		return &Ops{
			fwddesc:      fwddesc,
			bwdddesc:     bwdddesc,
			bwdfdesc:     bwdfdesc,
			perfforward:  input.FwdAlgo,
			perfbackdata: input.BwdDataAlgo,
			perfbackfilt: input.BwdFiltAlgo,
			fwdgroup:     input.FwdGroup,
			bwddgroup:    input.BwdDataGroup,
			bwdfgroup:    input.BwdFiltGroup,
			stride:       input.Stride,
			dilation:     input.Dilation,
			pad:          input.Pad,
		}, nil
	}
	fwddesc, err := helper.NewConvolutionNdDescriptor(input.CMode, input.Dtype, input.Pad, input.Stride, input.Dilation)
	if err != nil {
		return nil, err
	}
	bwdddesc, err := helper.NewConvolutionNdDescriptor(input.CMode, input.Dtype, input.Pad, input.Stride, input.Dilation)
	if err != nil {
		return nil, err
	}

	bwdfdesc, err := helper.NewConvolutionNdDescriptor(input.CMode, input.Dtype, input.Pad, input.Stride, input.Dilation)
	if err != nil {
		return nil, err
	}
	fwddesc.SetMathType(input.FwdAlgo.MathType)
	bwdfdesc.SetMathType(input.BwdFiltAlgo.MathType)
	bwdddesc.SetMathType(input.BwdDataAlgo.MathType)
	return &Ops{
		fwddesc:      fwddesc,
		bwdddesc:     bwdddesc,
		bwdfdesc:     bwdfdesc,
		perfforward:  input.FwdAlgo,
		perfbackdata: input.BwdDataAlgo,
		perfbackfilt: input.BwdFiltAlgo,
		fwdgroup:     input.FwdGroup,
		bwddgroup:    input.BwdDataGroup,
		bwdfgroup:    input.BwdFiltGroup,
		stride:       input.Stride,
		dilation:     input.Dilation,
		pad:          input.Pad,
	}, nil
}

//Info returns an info struct and error.  Info is usually used for saving the data to a json file.
func (c *Ops) Info() (Info, error) {
	mode, dtype, pad, stride, dilation, err := c.fwddesc.GetDescriptor()
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
