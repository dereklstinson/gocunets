package gocunets

//
///*
//
//type layer struct {
//	name         string
//	activation   *activation.Layer
//	cnn          *cnn.Layer
//	fcnn         *fcnn.Layer
//	softmax      *softmax.Layer
//	pool         *pooling.Layer
//	drop         *dropout.Layer
//	batch        *batchnorm.Layer
//	reshape      *reshape.Layer
//	cnntranspose *cnntranspose.Layer
//}
//
//*/
//
////BatchNorms returns the batchnorm layers in the network if nil is returned none are found
//func (m *Network) BatchNorms() []*batchnorm.Layer {
//	x := make([]*batchnorm.Layer, 0)
//	for i := range m.layers {
//		if m.layers[i].batch != nil {
//			x = append(x, m.layers[i].batch)
//		}
//	}
//	if len(x) == 0 {
//		return nil
//	}
//	return x
//}
//
////Dropouts returns the dropout layers in the network if nil is returned none are found
//func (m *Network) Dropouts() []*dropout.Layer {
//	x := make([]*dropout.Layer, 0)
//	for i := range m.layers {
//		if m.layers[i].drop != nil {
//			x = append(x, m.layers[i].drop)
//		}
//	}
//	if len(x) == 0 {
//		return nil
//	}
//	return x
//}
//
////Transposes returns the tranpose convolution layers in the network if nil is returned none are found
//func (m *Network) Transposes() []*cnntranspose.Layer {
//	x := make([]*cnntranspose.Layer, 0)
//	for i := range m.layers {
//		if m.layers[i].cnntranspose != nil {
//			x = append(x, m.layers[i].cnntranspose)
//		}
//	}
//	if len(x) == 0 {
//		return nil
//	}
//	return x
//}
//
////Convolutions returns the convolution layers in the network if nil is returned none are found
//func (m *Network) Convolutions() []*cnn.Layer {
//	x := make([]*cnn.Layer, 0)
//	for i := range m.layers {
//		if m.layers[i].cnn != nil {
//			x = append(x, m.layers[i].cnn)
//		}
//	}
//	if len(x) == 0 {
//		return nil
//	}
//	return x
//}
//
////Activations returns the activation layers in the network if nil is returned none are found
//func (m *Network) Activations() []*activation.Layer {
//	x := make([]*activation.Layer, 0)
//	for i := range m.layers {
//		if m.layers[i].activation != nil {
//			x = append(x, m.layers[i].activation)
//		}
//	}
//	if len(x) == 0 {
//		return nil
//	}
//	return x
//}
//
