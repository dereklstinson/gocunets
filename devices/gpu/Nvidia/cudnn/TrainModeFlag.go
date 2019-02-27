package cudnn

import (
	"github.com/dereklstinson/GoCudnn/xtra"
)

//TrainMode is a flag to set the TrainingMode
type TrainMode xtra.TrainingMode

//Cu is used to pass gocudnn.TrainMode flags for the functions inside the different cudnn packages
func (t TrainMode) Cu() xtra.TrainingMode {
	return xtra.TrainingMode(t)
}

//TrainModeFlag is used to pass TrainingMode flags through methods.
type TrainModeFlag struct {
	c xtra.TrainingModeFlag
}

//Adam is used to set the TrainingMode to Adam
func (t TrainModeFlag) Adam() TrainMode {
	return TrainMode(t.c.Adam())
}

//AdaDelta is used to set the TrainingMode to AdaDelta
func (t TrainModeFlag) AdaDelta() TrainMode {
	return TrainMode(t.c.AdaDelta())
}

//AdaGrad is used to set the TrainingMode to AdaGrad
func (t TrainModeFlag) AdaGrad() TrainMode {
	return TrainMode(t.c.AdaGrad())
}
