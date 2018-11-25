package reduce

import (
	"github.com/dereklstinson/GoCudnn"
)

//TypeMode is used for flags
type TypeMode gocudnn.IndiciesType

func (i TypeMode) cu() gocudnn.IndiciesType { return gocudnn.IndiciesType(i) }

//IndTypeFlag returns TypeMode flags
type IndTypeFlag struct {
	ind gocudnn.IndiciesTypeFlag
}

//Type32Bit returns  IndiciesType( C.CUDNN_32BIT_INDICES) flag
func (i IndTypeFlag) Type32Bit() TypeMode {
	return TypeMode(i.ind.Type32Bit())
}

//Type64Bit returns  IndiciesType( C.CUDNN_64BIT_INDICES) flag
func (i IndTypeFlag) Type64Bit() TypeMode {
	return TypeMode(i.ind.Type64Bit())
}

//Type16Bit returns IndiciesType( C.CUDNN_16BIT_INDICES) flag
func (i IndTypeFlag) Type16Bit() TypeMode {
	return TypeMode(i.ind.Type16Bit())
}

//Type8Bit returns  IndiciesType( C.CUDNN_8BIT_INDICES) flag
func (i IndTypeFlag) Type8Bit() TypeMode {
	return TypeMode(i.ind.Type8Bit())
}
