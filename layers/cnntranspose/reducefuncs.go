package cnntranspose

import "github.com/dereklstinson/GoCuNets/cudnn"

//SetupWStatReducers sets up the minmax reducers for the bias and weights
func (l *Layer) SetupWStatReducers(handle *cudnn.Handler) (err error) {
	return l.conv.SetupWStatReducers(handle)
}

//SetupDWStatReducers sets up the Delta W stat reducers for the bias and weights
func (l *Layer) SetupDWStatReducers(handle *cudnn.Handler) (err error) {
	return l.conv.SetupDWStatReducers(handle)
}

/*

Weights

*/

//WMax returns the Max weight value for the layer.
func (l *Layer) WMax(handle *cudnn.Handler) (float32, error) {
	return l.conv.WMax(handle)
}

//WMin returns the Min weight value for the layer
func (l *Layer) WMin(handle *cudnn.Handler) (float32, error) {
	return l.conv.WMin(handle)
}

// WAvg returns the avg weight value for the layer
func (l *Layer) WAvg(handle *cudnn.Handler) (float32, error) {
	return l.conv.WAvg(handle)
}

// WNorm1 returns the norm1 weight value for the layer
func (l *Layer) WNorm1(handle *cudnn.Handler) (float32, error) {
	return l.conv.WNorm1(handle)
}

// WNorm2 returns the norm2 weight value for the layer
func (l *Layer) WNorm2(handle *cudnn.Handler) (float32, error) {
	return l.conv.WNorm2(handle)
}

/*

Bias

*/

//BMax returns the Max bias value for the layer
func (l *Layer) BMax(handle *cudnn.Handler) (float32, error) {
	return l.conv.BMax(handle)
}

//BMin returns the Min bias value for the layer
func (l *Layer) BMin(handle *cudnn.Handler) (float32, error) {
	return l.conv.BMin(handle)
}

// BAvg returns the avg weight value for the layer
func (l *Layer) BAvg(handle *cudnn.Handler) (float32, error) {
	return l.conv.BAvg(handle)
}

// BNorm1 returns the norm1 bias value for the layer
func (l *Layer) BNorm1(handle *cudnn.Handler) (float32, error) {
	return l.conv.BNorm1(handle)
}

// BNorm2 returns the norm2 bias value for the layer
func (l *Layer) BNorm2(handle *cudnn.Handler) (float32, error) {
	return l.conv.BNorm2(handle)
}

/*

Delta Weights

*/

//DWMax returns the Max delta weight value for the layer
func (l *Layer) DWMax(handle *cudnn.Handler) (float32, error) {
	return l.conv.DWMax(handle)
}

//DWMin returns the Min delta weight value for the layer
func (l *Layer) DWMin(handle *cudnn.Handler) (float32, error) {
	return l.conv.DWMin(handle)
}

// DWAvg returns the avg delta weight value for the layer
func (l *Layer) DWAvg(handle *cudnn.Handler) (float32, error) {
	return l.conv.DWAvg(handle)
}

// DWNorm1 returns the norm1 delta weight value for the layer
func (l *Layer) DWNorm1(handle *cudnn.Handler) (float32, error) {
	return l.conv.DWNorm1(handle)
}

// DWNorm2 returns the norm2 delta weight value for the layer
func (l *Layer) DWNorm2(handle *cudnn.Handler) (float32, error) {
	return l.conv.DWNorm2(handle)
}

/*

Delta Bias

*/

//DBMax returns the Max delta bias value for the layer
func (l *Layer) DBMax(handle *cudnn.Handler) (float32, error) {
	return l.conv.DBMax(handle)
}

//DBMin returns the Min delta bias value for the layer
func (l *Layer) DBMin(handle *cudnn.Handler) (float32, error) {
	return l.conv.DBMin(handle)
}

// DBAvg returns the avg delta bias value for the layer
func (l *Layer) DBAvg(handle *cudnn.Handler) (float32, error) {
	return l.conv.DBAvg(handle)
}

// DBNorm1 returns the norm1 delta bias value for the layer
func (l *Layer) DBNorm1(handle *cudnn.Handler) (float32, error) {
	return l.conv.DBNorm1(handle)
}

// DBNorm2 returns the norm2 delta bias value for the layer
func (l *Layer) DBNorm2(handle *cudnn.Handler) (float32, error) {
	return l.conv.DBNorm2(handle)
}
