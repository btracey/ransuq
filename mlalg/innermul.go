package mlalg

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/reggo/reggo/common"
	"github.com/reggo/reggo/scale"
	"github.com/reggo/reggo/train"
)

func init() {
	common.Register(MulTrainer{})
}

// Multiplies the inner by a constant which is the first input
type MulTrainer struct {
	Inner train.Trainable
}

func (m MulTrainer) GrainSize() int {
	return m.Inner.GrainSize()
}

func (m MulTrainer) InputDim() int {
	return m.Inner.InputDim() + 1
}

func (m MulTrainer) OutputDim() int {
	return m.Inner.OutputDim()
}

func (m MulTrainer) NumFeatures() int {
	return m.Inner.NumFeatures() + 1
}

func (m MulTrainer) NumParameters() int {
	return m.Inner.NumParameters()
}

func (m MulTrainer) Parameters(s []float64) []float64 {
	return m.Inner.Parameters(s)
}

func (m MulTrainer) SetParameters(s []float64) {
	m.Inner.SetParameters(s)
}

func (m MulTrainer) RandomizeParameters() {
	m.Inner.RandomizeParameters()
}

func (m MulTrainer) NewFeaturizer() train.Featurizer {
	return mulFeaturizer{
		inner: m.Inner.NewFeaturizer(),
	}
}

func (m MulTrainer) Predictor() common.Predictor {
	return MulPredictor{
		inner: m.Inner.Predictor(),
	}
}

func (m MulTrainer) NewLossDeriver() train.LossDeriver {
	return mulTrainerLossDeriver{
		inner: m.Inner.NewLossDeriver(),
	}
}

type mulFeaturizer struct {
	inner train.Featurizer
}

func (m mulFeaturizer) Featurize(input, feature []float64) {
	if len(input) != len(feature) {
		panic("feature and input length mismatch")
	}

	m.inner.Featurize(input[1:], feature[1:])
	feature[0] = input[0]
}

type MulPredictor struct {
	inner common.Predictor
}

func (m MulPredictor) Predict(input, output []float64) ([]float64, error) {
	var err error
	output, err = m.inner.Predict(input[1:], output)
	if err != nil {
		return nil, err
	}
	for i := range output {
		output[i] *= input[0]
	}
	return output, nil
}

func (m MulPredictor) PredictBatch(inputs common.RowMatrix, outputs common.MutableRowMatrix) (common.MutableRowMatrix, error) {
	var err error
	outputs, err = m.inner.PredictBatch(inputs, outputs)
	if err != nil {
		return nil, err
	}
	nSamples, inputDim := outputs.Dims()
	for i := 0; i < nSamples; i++ {
		for j := 0; j < inputDim; j++ {
			v := outputs.At(i, j)
			v *= inputs.At(i, 0)
			outputs.Set(i, j, v)
		}
	}
	return outputs, nil
}

func (m MulPredictor) InputDim() int {
	return m.inner.InputDim() + 1
}

func (m MulPredictor) OutputDim() int {
	return m.inner.OutputDim()
}

type mulTrainerLossDeriver struct {
	inner train.LossDeriver
}

func (m mulTrainerLossDeriver) Predict(parameters, featurizedInput, predOutput []float64) {
	m.inner.Predict(parameters, featurizedInput[1:], predOutput)

	for i := range predOutput {
		predOutput[i] *= featurizedInput[0]
	}
}

func (m mulTrainerLossDeriver) Deriv(parameters, featurizedInput, predOutput, dLossDPred, dLossDWeight []float64) {
	// The inner's predicted output was divided by the input
	for i := range predOutput {
		predOutput[i] /= featurizedInput[0]
	}

	m.inner.Deriv(parameters, featurizedInput[1:], predOutput, dLossDPred, dLossDWeight)

	for i := range predOutput {
		predOutput[i] *= featurizedInput[0]
	}

	// Need to multiply the derivatives by a
	for i := range dLossDWeight {
		dLossDWeight[i] *= featurizedInput[0]
	}
}

type MulScaler struct {
	Scaler   scale.Scaler
	mulScale float64
}

// Don't scale the first input

func (m *MulScaler) SetScale(data *mat64.Dense) error {
	// Get a view of the data without the first column
	r, c := data.Dims()
	mat := &mat64.Dense{}
	mat.View(data, 0, 1, r, c-1)

	// Set the scale such that the sum of the scales is 1
	maxval := math.Inf(-1)
	for i := 0; i < r; i++ {
		v := data.At(i, 0)
		if v < 0 {
			panic("This assumes positive scale")
		}
		if v > maxval {
			maxval = v
		}

	}

	fmt.Println("mul scale is ", maxval)

	m.mulScale = maxval

	return m.Scaler.SetScale(mat)
}

func (m MulScaler) Scale(point []float64) error {
	point[0] /= m.mulScale
	return m.Scaler.Scale(point[1:])
}

func (m MulScaler) Unscale(point []float64) error {
	point[0] *= m.mulScale
	return m.Scaler.Unscale(point[1:])
}

func (m MulScaler) IsScaled() bool {
	return m.Scaler.IsScaled()
}

func (m MulScaler) Dimensions() int {
	return m.Scaler.Dimensions()
}
