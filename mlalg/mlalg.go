package mlalg

/*
type ExtraFunc struct {
	// Modifies the base prediction in place
	Prediction func(extraInputs, basePrediction, newPrediction []float64)
	//
	Derivative func(extraInputs, basePrediction, newPrediction []float64)
	NumExtra   int
}
*/

/*

// Extra trainer is an ML alg who is a composition of the inner prediction
// Inner one uses the first n inputs. Outer one uses all of the inputs and the
// outer one uses the output of the inner plus all of the inputs
// THIS CODE DOESN"T WORK
type ExtraTrainer struct {
	Inner       train.Trainable
	Outer       train.Trainable
	NumInputs   int
	InnerInputs []int
	OuterInputs []int // Which inputs go to both inner and outer (in addition to the outputs from the inner)
}

// Check that the sizes are right
func (l ExtraTrainer) Check() error {
	if len(l.InnerInputs) != l.Inner.InputDim() {
		return errors.New("Input dimension does not match for inner")
	}
	if len(l.OuterInputs)+l.Inner.OutputDim() != l.Outer.InputDim() {
		return errors.New("Input dimension does not match for outer")
	}

	// TODO: Check the inds to see that they don't overreach
}

// From the full set of inputs, extract the inner input
func setInnerInput(inputs []float64, innerInputs []int, inner []float64) {
	if len(inner) != len(innerInputs) {
		panic("wrong size inner input")
	}
	for i, idx := range innerInputs {
		inner[i] = inputs[idx]
	}
}

//
func setOuterInput(inputs []float64, outerInputs []int, inneroutput, outer []float64) {
	for i, idx := range outerInputs {
		outer[i] = inputs[idx]
	}
	// Add to the outer
	copy(outer[len(outerInputs):], inneroutput)
}

func (l ExtraTrainer) GrainSize() int {
	innerSize := l.Inner.GrainSize()
	outerSize := l.Outer.GrainSize()
	if innerSize < outerSize {
		return innerSize
	}
	return outerSize
}

func (l ExtraTrainer) InputDim() int {
	return l.Outer.InputDim()
}

func (l ExtraTrainer) OutputDim() int {
	return l.Outer.OutputDim()
}

// TODO: Need to do something about featurizer. Weird if allow input overlap
/*
func (l ExtraTrainer) NewFeaturizer() train.Featurizer {
	return featurizer{
		inner: l.Base.NewFeaturizer(),
		outer: l.Outer.NewFeaturizer(),
	}
}
*/

/*
func (l ExtraTrainer) NumFeatures() int {
	return l.Outer.NumFeatures()
}

func (l ExtraTrainer) NewLossDeriver() train.LossDeriver {
	return lossDeriver{
		inner: l.Inner.NewLossDeriver(),
		outer: l.Outer.NewLossDeriver(),
	}
}

func (l ExtraTrainer) NumParameters() int {
	return l.Inner.NumParameters() + l.Outer.NumParameters()
}

func (l ExtraTrainer) Parameters(s []float64) []float64 {
	nTotalParam = l.Inner.NumParameters() + l.Outer.NumParameters()
	if len(s) < nTotalParam {
		s = make([]float64, nTotalParam)
	} else {
		s = s[:nTotalParam]
	}
	l.Inner.Parameters(s[:l.Inner.NumParameters()])
	l.Outer.Parameters(s[l.Inner.NumParameters():l.Outer.NumParameters()])
	return s
}

func (l ExtraTrainer) RandomizeParameters() {
	l.Inner.RandomizeParameters()
	l.Outer.RandomizeParameters()
}

func (l ExtraTrainer) SetParameters(s []float64) {
	inner := l.Inner.NumParameters()
	outer := l.Outer.NumParameters()

	if len(s) != inner+outer {
		panic("bad size")
	}

	l.Inner.SetParameters(s[:inner])
	l.Outer.SetParameters(s[inner:outer])
}

func (l ExtraTrainer) Predictor() common.Predictor {
	return predictor{
		inner: l.Inner.Predictor(),
		outer: l.Outer.Predictor(),
		tmp:   make([]float64, l.Inner.OutputDim()),
	}
}

type predictor struct {
	inner           common.Predictor
	outer           common.Predictor
	innerInputTmp   []float64
	innerOutputTmp  []float64
	outerInputTmp   []float64
	innerInputSlice []int
	outerInputSlice []int
}

func (p predictor) Predict(input, output []float64) {
	// First predict on the first number of inputs
	setInnerInput(inputs, p.innerInputSlice, p.innerInputTmp)
	p.inner.Predict(p.innerInputSlice, p.innerOutputTmp)

	// Now predict on the outer
	setOuterInput(inputs, p.outerInputSlice, p.innerOutputTmp, p.outerInputTmp)
	p.outer.Predict(p.outerInputTmp, output)
}

type featurizer struct {
	inner  train.Featurizer
	output train.Featurizer
}

/*
func (f featurizer) Featurize(input, feature []float64) {
	// Keep the first input the same, and then the rest on whatever the nnet does
	copy(feature[:f.extraInputs], input[:f.extraInputs])
	f.base.Featurize(input[f.extraInputs:], feature[f.extraInputs:])
}
*/
/*
type lossDeriver struct {
	inner           train.LossDeriver
	outer           train.LossDeriver
	innerInputTmp   []float64
	innerOutputTmp  []float64
	outerInputTmp   []float64
	innerInputSlice []int
	outerInputSlice []int
}

func (l lossDeriver) Deriv(parameters, featurizedInput, predOutput, dLossDPred, dLossDWeight []float64) {

	// Computes the derivative of the loss function with respect to the parameters

	// The extra inputs don't affect it at all beyond their influence on dLossDPred
	// TODO: Is this true?
	l.base.Deriv(parameters, featurizedInput, predOutput, dLossDPred, dLossDWeight)
}

func (l lossDeriver) Predict(parameters, featurizedInput, predOutput []float64) {
	l.base.Predict(parameters, featurizedInput[l.extraInputs:], predOutput)
}

// Featurizer need to do something tricky

var t = train.Trainable(ExtraTrainer{})

// Shoot, need to hack the output
type MulLoss struct {
}
*/
