package reduce

import (
	"github.com/dereklstinson/GoCudnn"
)

//IndiciesMode ris used for glags
type IndiciesMode gocudnn.ReduceTensorIndices

//IndiciesFLag returns indiciciesmode
type IndiciesFLag struct {
	flg gocudnn.ReduceTensorIndicesFlag
}

func (r IndiciesMode) cu() gocudnn.ReduceTensorIndices {
	return gocudnn.ReduceTensorIndices(r)
}

//NoIndices returns IndiciesMode
func (r IndiciesFLag) NoIndices() IndiciesMode {
	return IndiciesMode(r.flg.NoIndices())
}

//FlattenedIndicies returns IndiciesMode
func (r IndiciesFLag) FlattenedIndicies() IndiciesMode {
	return IndiciesMode(r.flg.FlattenedIndicies())
}
