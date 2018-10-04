package fcnn

//Settings contains the settings needed to build NeuralNetwork
type Settings struct {
	Outputs int32 `json:"outputs,omitempty"`
	Managed bool  `json:"managed,omitempty"`
}
