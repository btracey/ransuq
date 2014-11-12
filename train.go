package ransuq

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"runtime"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize"

	"github.com/reggo/reggo/common"
	"github.com/reggo/reggo/loss"
	"github.com/reggo/reggo/regularize"
	"github.com/reggo/reggo/scale"
	regtrain "github.com/reggo/reggo/train"
)

type Predictor interface {
	common.Predictor
}

//var _ =      floats.Norm(s, L)

type Trainable interface {
	regtrain.Trainable
}

type Trainer struct {
	TrainSettings
	InputScaler  scale.Scaler
	OutputScaler scale.Scaler
	Losser       loss.DerivLosser
	Regularizer  regularize.Regularizer
	Algorithm    Trainable
}

// TODO: This should be implemented in Reggo

// Fit is the data that has been trained (and can be saved etc.)
type ScalePredictor struct {
	Predictor
	InputScaler  scale.Scaler
	OutputScaler scale.Scaler
}

func (s *ScalePredictor) Predict(input, output []float64) ([]float64, error) {
	s.InputScaler.Scale(input)
	var err error
	output, err = s.Predictor.Predict(input, output)
	s.InputScaler.Unscale(input)
	s.OutputScaler.Unscale(output)
	return output, err
}

func (s *ScalePredictor) MarshalJSON() ([]byte, error) {
	// Make a couple of random inputs
	nRand := 5
	inputs := make([][]float64, nRand)
	outputs := make([][]float64, nRand)
	for i := range inputs {
		inputs[i] = make([]float64, s.Predictor.InputDim())
		for j := range inputs[i] {
			inputs[i][j] = rand.NormFloat64()
		}
		outputs[i] = make([]float64, s.Predictor.OutputDim())
		// Inputs are already scaled. May need to change this for different scalers
		s.Predictor.Predict(inputs[i], outputs[i])
		s.InputScaler.Unscale(inputs[i])
		s.OutputScaler.Unscale(outputs[i])
	}

	// Save the algorithm as well as the scalares
	saveStruct := struct {
		Predictor    common.InterfaceMarshaler
		InputScaler  common.InterfaceMarshaler
		OutputScaler common.InterfaceMarshaler
		TestInputs   [][]float64
		TestOutputs  [][]float64
	}{
		Predictor:    common.InterfaceMarshaler{I: s.Predictor},
		InputScaler:  common.InterfaceMarshaler{I: s.InputScaler},
		OutputScaler: common.InterfaceMarshaler{I: s.OutputScaler},
		TestInputs:   inputs,
		TestOutputs:  outputs,
	}

	return json.Marshal(saveStruct)
}

func (s *ScalePredictor) UnmarshalJSON(b []byte) error {
	saveStruct := struct {
		Predictor    common.InterfaceMarshaler
		InputScaler  common.InterfaceMarshaler
		OutputScaler common.InterfaceMarshaler
		TestInputs   [][]float64
		TestOutputs  [][]float64
	}{}
	err := json.Unmarshal(b, &saveStruct)
	if err != nil {
		return err
	}
	s.Predictor = saveStruct.Predictor.I.(common.Predictor)
	s.InputScaler = saveStruct.InputScaler.I.(scale.Scaler)
	s.OutputScaler = saveStruct.OutputScaler.I.(scale.Scaler)

	// Test that the predictions match
	inputs := saveStruct.TestInputs
	outputs := saveStruct.TestOutputs

	testOutputs := make([][]float64, len(outputs))
	for i := range inputs {
		s.InputScaler.Scale(inputs[i])
		testOutputs[i] = make([]float64, len(outputs[i]))
		s.Predictor.Predict(inputs[i], testOutputs[i])
		s.OutputScaler.Unscale(testOutputs[i])
		if !floats.EqualApprox(testOutputs[i], outputs[i], 1e-13) {
			return fmt.Errorf("prediction didn't match stored output. Found %v, expected %v", testOutputs[i], outputs[i])
		}
	}
	return nil
}

// TODO: Make weights a vector (interface rather than explicit data)

type TrainResults struct {
	OptObj              float64
	OptGradNorm         float64
	FunctionEvaluations int
}

// Train trains the algorithm returning a predictor
func (t *Trainer) Train(inputs, outputs common.RowMatrix, weights []float64) (Predictor, TrainResults, error) {

	inputScaler := t.InputScaler
	outputScaler := t.OutputScaler
	losser := t.Losser
	regularizer := t.Regularizer
	algorithm := t.Algorithm

	emptyResults := TrainResults{}

	// Set the scale
	// TODO: Need to fix reggo/scale such that can use MutableRowMatrix etc.
	if inputScaler != nil {
		iDense := inputs.(*mat64.Dense)
		inputScaler.SetScale(iDense)
	}

	if outputScaler != nil {
		oDense := outputs.(*mat64.Dense)
		outputScaler.SetScale(oDense)
	}

	if inputScaler != nil {
		iDense := inputs.(*mat64.Dense)
		scale.ScaleData(inputScaler, iDense)
		defer scale.UnscaleData(inputScaler, iDense)
	}
	if outputScaler != nil {
		oDense := outputs.(*mat64.Dense)
		scale.ScaleData(outputScaler, oDense)
		defer scale.UnscaleData(outputScaler, oDense)
	}

	// Train the algorithm

	// Check the algorithm can be trained with a linear solve
	if regtrain.CanLinearSolve(algorithm, losser, regularizer) {
		linearTrainable := algorithm.(regtrain.LinearTrainable)

		fmt.Println("In linear solve")
		parameters := regtrain.LinearSolve(linearTrainable, nil, inputs, outputs, weights, regularizer)
		if parameters == nil {
			return nil, emptyResults, fmt.Errorf("mldriver: error during linear solve")
		}
		algorithm.SetParameters(parameters)
		return algorithm.Predictor(), emptyResults, nil
	}

	fmt.Println("starting algorithm training")
	algorithm.RandomizeParameters()
	param := algorithm.Parameters(nil)

	fmt.Println("before newbatch")
	fmt.Println("losser is ", losser)

	// Create the trainer
	problem := &regtrain.GradOptimizable{
		Trainable: algorithm,
		Inputs:    inputs,
		Outputs:   outputs,
		Weights:   weights,

		NumWorkers:  runtime.GOMAXPROCS(0),
		Losser:      losser,
		Regularizer: regularizer,
	}

	settings := optimize.DefaultSettings()
	settings.FunctionEvals = t.TrainSettings.MaxFunEvals
	settings.GradientAbsTol = t.TrainSettings.GradAbsTol
	settings.FunctionAbsTol = t.TrainSettings.ObjAbsTol

	problem.Init()
	result, err := optimize.Local(problem, param, settings, nil)
	if err != nil {
		return nil, emptyResults, err
	}
	problem.Close()

	emptyResults.FunctionEvaluations = result.FunctionEvals
	emptyResults.OptGradNorm = result.GradientNorm
	emptyResults.OptObj = result.F
	algorithm.SetParameters(result.X)

	/*
		// Create the trainer
		batch := regtrain.NewBatchGradBased(algorithm, true, inputs, outputs, weights, losser, regularizer)
		problem := batch
		optsettings := multivariate.DefaultSettings()
		optsettings.GradAbsTol = t.TrainSettings.GradAbsTol
		optsettings.ObjAbsTol = t.TrainSettings.ObjAbsTol
		optsettings.MaximumFunctionEvaluations = t.TrainSettings.MaxFunEvals
		// Optimize the results
		result, err := multivariate.OptimizeGrad(problem, param, optsettings, nil)
		if err != nil {
			return nil, emptyResults, err
		}

		emptyResults.FunctionEvaluations = result.FunctionEvaluations
		emptyResults.OptGradNorm = floats.Norm(result.Grad, 2)
		emptyResults.OptObj = result.Obj
		algorithm.SetParameters(result.Loc)
	*/
	//
	//
	//
	regpred := algorithm.Predictor()

	// cast predictor as a predictor
	pred, ok := regpred.(Predictor)
	if !ok {
		return nil, emptyResults, errors.New("predictor is not a Predictor")
	}

	sp := &ScalePredictor{
		Predictor:    pred,
		InputScaler:  t.InputScaler,
		OutputScaler: t.OutputScaler,
	}

	return sp, emptyResults, nil
}
