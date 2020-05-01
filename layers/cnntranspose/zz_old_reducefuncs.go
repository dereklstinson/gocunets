package cnntranspose

/*
import (
	"errors"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
)

//
////SetupWStatReducers sets up the minmax reducers for the bias and weights
//func (l *Layer) SetupWStatReducers(handle *cudnn.Handler) (err error) {
//	err = l.conv.SetupWStatReducers(handle)
//
//	return err
//}
//
////SetupDWStatReducers sets up the Delta W stat reducers for the bias and weights
//func (l *Layer) SetupDWStatReducers(handle *cudnn.Handler) (err error) {
//	return l.conv.SetupDWStatReducers(handle)
//}
//
//
//
//Weights
//
//

//WMax returns the Max weight value for the layer.
func (l *Layer) WMax(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.WMax(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: WMax: " + err.Error())
	}
	return x, nil
}

//WMin returns the Min weight value for the layer
func (l *Layer) WMin(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.WMin(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: WMin: " + err.Error())
	}
	return x, nil

}

// WAvg returns the avg weight value for the layer
func (l *Layer) WAvg(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.WAvg(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: WAvg: " + err.Error())
	}
	return x, nil

}

// WNorm1 returns the norm1 weight value for the layer
func (l *Layer) WNorm1(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.WNorm1(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: WNorm1: " + err.Error())
	}
	return x, nil

}

// WNorm2 returns the norm2 weight value for the layer
func (l *Layer) WNorm2(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.WNorm2(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: WNorm2: " + err.Error())
	}
	return x, nil

}


//
//Bias
//


//BMax returns the Max bias value for the layer
func (l *Layer) BMax(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.BMax(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: BMax: " + err.Error())
	}
	return x, nil

}

//BMin returns the Min bias value for the layer
func (l *Layer) BMin(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.BMin(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: BMin: " + err.Error())
	}
	return x, nil

}

// BAvg returns the avg weight value for the layer
func (l *Layer) BAvg(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.BAvg(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: BAvg: " + err.Error())
	}
	return x, nil

}

// BNorm1 returns the norm1 bias value for the layer
func (l *Layer) BNorm1(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.BNorm1(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: BNorm1: " + err.Error())
	}
	return x, nil

}

// BNorm2 returns the norm2 bias value for the layer
func (l *Layer) BNorm2(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.BNorm2(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: BNorm2: " + err.Error())
	}
	return x, nil

}


//
//Delta Weights
//


//DWMax returns the Max delta weight value for the layer
func (l *Layer) DWMax(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.DWMax(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: DWMax: " + err.Error())
	}
	return x, nil

}

//DWMin returns the Min delta weight value for the layer
func (l *Layer) DWMin(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.DWMin(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: DWMin: " + err.Error())
	}
	return x, nil

}

// DWAvg returns the avg delta weight value for the layer
func (l *Layer) DWAvg(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.DWAvg(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: DWAvg: " + err.Error())
	}
	return x, nil

}

// DWNorm1 returns the norm1 delta weight value for the layer
func (l *Layer) DWNorm1(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.DWNorm1(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: DWNorm1: " + err.Error())
	}
	return x, nil

}

// DWNorm2 returns the norm2 delta weight value for the layer
func (l *Layer) DWNorm2(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.DWNorm2(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: DWNorm2: " + err.Error())
	}
	return x, nil

}


//
//Delta Bias
//


//DBMax returns the Max delta bias value for the layer
func (l *Layer) DBMax(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.DBMax(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: DBMax: " + err.Error())
	}
	return x, nil

}

//DBMin returns the Min delta bias value for the layer
func (l *Layer) DBMin(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.DBMin(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: DBMin: " + err.Error())
	}
	return x, nil

}

// DBAvg returns the avg delta bias value for the layer
func (l *Layer) DBAvg(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.DBAvg(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: DBAvg: " + err.Error())
	}
	return x, nil

}

// DBNorm1 returns the norm1 delta bias value for the layer
func (l *Layer) DBNorm1(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.DBNorm1(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: DBNorm1: " + err.Error())
	}
	return x, nil

}

// DBNorm2 returns the norm2 delta bias value for the layer
func (l *Layer) DBNorm2(handle *cudnn.Handler) (float32, error) {
	x, err := l.conv.DBNorm2(handle)
	if err != nil {
		return 0, errors.New("CnnTranspose: DBNorm2: " + err.Error())
	}
	return x, nil

}
*/
