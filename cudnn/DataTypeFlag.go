package cudnn

import gocudnn "github.com/dereklstinson/GoCudnn"

//DataType is used for flags to select what datatype is wanted
type DataType gocudnn.DataType

//DataTypeFlag is used to pass DataType flags through methods
type DataTypeFlag struct {
	c gocudnn.DataTypeFlag
}

//Float returns the Float flag
func (d DataTypeFlag) Float() DataType {
	return DataType(d.c.Float())
}

//Double returns the Double flag
func (d DataTypeFlag) Double() DataType {
	return DataType(d.c.Double())
}

//Int32 returns the Int32 flag
func (d DataTypeFlag) Int32() DataType {
	return DataType(d.c.Int32())
}

//UInt8 return the UInt8 flag
func (d DataTypeFlag) UInt8() DataType {
	return DataType(d.c.UInt8())
}

//Int8 returns the Int8 flag
func (d DataTypeFlag) Int8() DataType {
	return DataType(d.c.Int8())
}

//Cu returns the gocudnn.DataType Flag.
func (d DataType) Cu() gocudnn.DataType {
	return gocudnn.DataType(d)
}
