package activation

/*
//UpdateCoef will update the coef.  ALthough it will change the descriptor if the activation mode doesn't use the coef scalar then it won't do anything.
func (a *Layer) UpdateCoef(coef float64) error {
	amode, nanprop, _, err := a.act.Properties()
	if err != nil {
		return err
	}
	return a.act.ReStage(amode, nanprop, coef)
}

//UpdateMode will update the mode
func (a *Layer) UpdateMode(amode gocudnn.ActivationMode) error {
	_, nanprop, coef, err := a.act.Properties()
	if err != nil {
		return err
	}

	return a.act.ReStage(amode, nanprop, coef)
}

//NotPropigateNAN sets up the layer to not propigate nan values
func (a *Layer) NotPropigateNAN() error {
	if a.nanproped == gocudnn.PropagationNAN(0) {
		return nil
	}
	a.nanproped = gocudnn.PropagationNAN(0)
	return a.updatenanprop()

}

//PropigateNAN sets up the layer to propigate nan values
func (a *Layer) PropigateNAN() error {
	if a.nanproped == gocudnn.PropagationNAN(1) {
		return nil
	}
	a.nanproped = gocudnn.PropagationNAN(1)
	return a.updatenanprop()

}

func (a *Layer) updatenanprop() error {
	amode, _, coef, err := a.act.Properties()
	if err != nil {
		return err
	}
	return a.act.ReStage(amode, a.nanproped, coef)
}
*/
