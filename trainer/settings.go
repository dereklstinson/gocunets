package trainer

//Settings contains the settings of a trainer
type Settings struct {
	Beta1    float64 `json:"beta_1,omitempty"`
	Beta2    float64 `json:"beta_2,omitempty"`
	Decay1   float64 `json:"decay_1,omitempty"`
	Decay2   float64 `json:"decay_2,omitempty"`
	Rate     float64 `json:"rate,omitempty"`
	Momentum float64 `json:"momentum,omitempty"`
	Eps      float64 `json:"eps,omitempty"`
	Batch    float64 `json:"batch,omitempty"`
	Managed  bool    `json:"managed,omitempty"`
}

//TSettings contains the trainer settings per trainer
type TSettings struct {
	Adam     Settings
	Momentum Settings
}
