package mlalg

import (
	"encoding/json"
	"errors"
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/reggo/reggo/common"
	"github.com/reggo/reggo/scale"
	"github.com/reggo/reggo/train"
)

func init() {
	common.Register(MulPredictor{})
	common.Register(&MulInputScaler{})
	common.Register(&MulOutputScaler{})
}

// Multiplies the inner by a constant which is the first input
// This should be used as the output scaler
type MulTrainer struct {
	Inner train.Trainable
}

// MAKE THIS AN INTERFACE

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
	//fmt.Println("In mul trainer set parameters", len(s))
	//fmt.Println("Mul trainer num parameters", m.NumParameters())
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

type mulPredictorMarshaler struct {
	Inner common.InterfaceMarshaler
}

func (m MulPredictor) MarshalJSON() ([]byte, error) {
	return json.Marshal(mulPredictorMarshaler{
		Inner: common.InterfaceMarshaler{
			I: m.inner,
		},
	})
}

func (m *MulPredictor) UnmarshalJSON(b []byte) error {
	v := mulPredictorMarshaler{}
	err := json.Unmarshal(b, &v)
	if err != nil {
		return err
	}
	m.inner = v.Inner.I.(common.Predictor)
	return nil
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

// Output scaler transforms the magnitude so that the average output is 1, but
// doesn't change the linear offset.
type MulOutputScaler struct {
	mulScale float64
	isScaled bool
}

func (m *MulOutputScaler) SetScale(data *mat64.Dense) error {
	// Get a view of the data without the first column
	r, c := data.Dims()

	if c != 1 {
		return errors.New("only one output dimenison allowed")
	}

	// Set the scale such that the average of the outputs is 1
	var sum float64
	for i := 0; i < r; i++ {
		v := data.At(i, 0)
		sum += math.Abs(v)
	}

	m.mulScale = sum / float64(r) // Old way that creates bad scaling
	//m.mulScale = sum
	//m.mulScale = sum / math.Sqrt(float64(r))
	//fmt.Println("mul scale = ", m.mulScale)
	m.isScaled = true

	return nil
}

func (m MulOutputScaler) Scale(point []float64) error {
	if len(point) != 1 {
		errors.New("only one output dimenison allowed")
	}
	point[0] /= m.mulScale
	return nil
}

func (m MulOutputScaler) Unscale(point []float64) error {
	if len(point) != 1 {
		errors.New("only one output dimenison allowed")
	}
	point[0] *= m.mulScale
	return nil
}

func (m MulOutputScaler) Dimensions() int {
	return 1
}

func (m MulOutputScaler) IsScaled() bool {
	return m.isScaled
}

func (m MulOutputScaler) Linear() (shift, scale []float64) {
	shift = []float64{0}
	scale = []float64{m.mulScale}
	return
}

type mulOutputMarshaler struct {
	IsScaled bool
	MulScale float64
}

func (m *MulOutputScaler) MarshalJSON() ([]byte, error) {
	return json.Marshal(mulOutputMarshaler{
		IsScaled: m.isScaled,
		MulScale: m.mulScale,
	})
}

func (m *MulOutputScaler) UnmarshalJSON(b []byte) error {
	v := &mulOutputMarshaler{}
	err := json.Unmarshal(b, &v)
	if err != nil {
		return err
	}
	m.isScaled = v.IsScaled
	m.mulScale = v.MulScale
	return nil
}

// needs the output scaler because we need to scale the multiplier if the output
// has been scaled
type MulInputScaler struct {
	Scaler scale.Scaler
	*MulOutputScaler
}

type mulInputMarshaler struct {
	Scaler          common.InterfaceMarshaler
	MulOutputScaler *MulOutputScaler
}

// Need to implement JSON because scaler is an interface
func (m *MulInputScaler) MarshalJSON() ([]byte, error) {
	return json.Marshal(mulInputMarshaler{
		Scaler: common.InterfaceMarshaler{
			I: m.Scaler,
		},
		MulOutputScaler: m.MulOutputScaler,
	})
}

func (m *MulInputScaler) UnmarshalJSON(b []byte) error {
	v := &mulInputMarshaler{}
	err := json.Unmarshal(b, &v)
	if err != nil {
		return err
	}
	m.Scaler = v.Scaler.I.(scale.Scaler)
	m.MulOutputScaler = v.MulOutputScaler
	return nil
}

// This relies on the OutputScaler being called (not necessarily first)
func (m *MulInputScaler) SetScale(data *mat64.Dense) error {
	// Get a view of the data without the first column
	r, c := data.Dims()
	mat := &mat64.Dense{}
	mat.View(data, 0, 1, r, c-1)

	return m.Scaler.SetScale(mat)
}

func (m MulInputScaler) Scale(point []float64) error {
	if m.MulOutputScaler.mulScale == 0 {
		panic("mulScale is zero")
	}
	point[0] /= m.MulOutputScaler.mulScale
	return m.Scaler.Scale(point[1:])
}

func (m MulInputScaler) Unscale(point []float64) error {
	point[0] *= m.MulOutputScaler.mulScale
	return m.Scaler.Unscale(point[1:])
}

func (m MulInputScaler) IsScaled() bool {
	return m.Scaler.IsScaled()
}

func (m MulInputScaler) Dimensions() int {
	return m.Scaler.Dimensions() + 1
}
