package ransuq

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	/*
		"github.com/reggo/reggo/common"
		"github.com/reggo/reggo/loss"
		"github.com/reggo/reggo/regularize"
		"github.com/reggo/reggo/scale"
	*/)

type GenerationError struct {
	Training []error
	Testing  []error
}

func (g GenerationError) Error() string {
	var str string
	if g.Training != nil {
		str += "training: "
		for i := range g.Training {
			str += fmt.Sprintf("error generating ", i, ": ", g.Training[i].Error())
		}
	}
	if g.Testing != nil {
		str += "testing: "
		for i := range g.Testing {
			str += fmt.Sprintf("error generating ", i, ": ", g.Testing[i].Error())
		}
	}
	return str
}

// MlTurb is the main script for learning and testing a ML model from data
func MlTurb(settings *Settings) error {
	// Do some error handling
	nTrainingSets := len(settings.TrainingData)
	if nTrainingSets == 0 {
		return errors.New("no training sets provided")
	}
	if len(settings.InputFeatures) == 0 {
		return errors.New("no input features")
	}
	if len(settings.OutputFeatures) == 0 {
		return errors.New("no output features")
	}

	// TODO: Add some fast path about checking for the existance of an algorithm
	// already (don't need to run training data, etc. if we already have a learned algorithm)

	// Generate the training data
	trainingDatasets := settings.TrainingData
	wgTrain := &sync.WaitGroup{}
	wgTrain.Add(1)
	var trainGenErr []error
	go func() {
		trainGenerators := GetGeneratables(trainingDatasets)
		trainGenErr = GenerateData(trainGenerators)
		trainGenErr = reduceError(trainGenErr)
		wgTrain.Done()
	}()
	// This should be removed later, but for now we have to do this because
	// we don't check to see if the same datasets are in testing and training
	wgTrain.Wait()

	// Generate the testing data
	testingDatasets := settings.TestingData

	wgTest := &sync.WaitGroup{}
	wgTest.Add(1)
	var testGenErr []error
	go func() {
		testGenerators := GetGeneratables(testingDatasets)
		testGenErr = GenerateData(testGenerators)
		testGenErr = reduceError(testGenErr)
		wgTest.Done()
	}()

	// Wait until all of the training runs are generated
	wgTrain.Wait()
	if trainGenErr != nil {
		// Can't go on if there is an error generating the training data
		// Wait until the testing data is generated so computation time isn't wasted
		wgTest.Wait()
		return GenerationError{
			Training: trainGenErr,
			Testing:  testGenErr,
		}
	}

	// Load all of the training data
	inputs, outputs, weights, loadErrs := LoadTrainingData(trainingDatasets, DenseLoad, settings.InputFeatures,
		settings.OutputFeatures, settings.WeightFeatures, settings.WeightFunc)

	if loadErrs != nil {
		// Can't continue on if there were errors loading training data
		// Wait until the testing data is generated so computation time isn't wasted
		wgTest.Wait()
		return loadErrs
	}

	nSamples, inputDim := inputs.Dims()
	outputSamples, outputDim := outputs.Dims()
	weightSamples := len(weights)

	// Quick assertions. These are panics because they should never happen
	if outputSamples != nSamples {
		panic("sample size mismatch")
	}
	if weightSamples != nSamples && weightSamples != 0 {
		panic("weight size mismatch")
	}
	if inputDim != len(settings.InputFeatures) {
		panic("input dim size mismatch")
	}
	if outputDim != len(settings.OutputFeatures) {
		panic("output dim size mismatch")
	}

	fmt.Println("NumSamples: ", nSamples)
	fmt.Println("InputDim: ", inputDim)
	fmt.Println("OutputDim: ", outputDim)
	fmt.Println()

	algsavepath := PredictorDirectory(settings.Savepath)

	err := os.MkdirAll(algsavepath, 0700)
	if err != nil {
		wgTest.Wait()
		return err
	}

	algFile := PredictorFilename(settings.Savepath)

	// See if the training algorithm exists
	_, err = os.Open(algFile)
	if err != nil && !os.IsNotExist(err) {
		// There is an error and it's something other than the file doesn't exist
		wgTest.Wait()
		return errors.New("error trying to open trained algorithm: " + err.Error())
	} else {

	}
	if os.IsNotExist(err) {
		// The algorithm does not exist
		// TODO: Replace this by making algorithm a "trainer" so can more easily incorporate
		// FANN et al
		trainer := settings.Trainer
		var sp Predictor

		sp, err := trainer.Train(inputs, outputs, weights)
		if err != nil {
			wgTest.Wait()
			return errors.New("error training: " + err.Error())
		}
		fmt.Println("Before save")
		err = savePredictor(sp, algFile)
		if err != nil {
			wgTest.Wait()
			return errors.New("error saving predictor: " + err.Error())
		}
		fmt.Println("Done saving")
	} else {
		fmt.Println("Algorithm already trained")
	}

	// Load the training algorithm
	scalePredictor, err := LoadScalePredictor(settings.Savepath)
	if err != nil {
		wgTest.Wait()
		return errors.New("error loading training algorithm: " + err.Error())
	}

	// Wait for all of the training algorithms to finish computing
	wgTest.Wait()

	_ = scalePredictor

	// Do post processing and running new machine-learning algorithms
	// Post process the testing data
	postWg := &sync.WaitGroup{}
	postWg.Add(1)
	var postErr error
	go func() {
		fmt.Println("Starting post-process")
		postErr = postprocess(scalePredictor, settings)
		fmt.Println("Done post-process")
		postWg.Done()
	}()

	// Loop over all the testing sets, see if they are comparers, and
	// run them if they are

	compWg := &sync.WaitGroup{}
	compWg.Add(1)
	var compErr []error
	go func() {
		// Get the generators
		var generators []Generatable
		for i := range testingDatasets {
			outLoc := filepath.Join(settings.Savepath, "comparison")
			comp, ok := testingDatasets[i].(Comparable)
			if ok {

				gen, err := comp.Comparison(algFile, outLoc, settings.FeatureSet)
				if err != nil {
					panic(err)
				}
				generators = append(generators, gen)
			}
		}
		compErr = GenerateData(generators)
		for _, gen := range generators {
			p, ok := gen.(PostProcessor)
			if ok {
				fmt.Println("Ok posprocessor")
				err = p.PostProcess()
				if err != nil {
					panic(err)
				}
			}
		}
		compWg.Done()
	}()
	postWg.Wait()
	compWg.Wait()
	compErr = reduceError(compErr)
	if compErr != nil {
		return err
	}
	if postErr != nil {
		return err
	}

	// Should do something about post-processing

	return nil
}

// reduceError returns the slice of errors if any of the errors are non-nil
// and nil if all of the errors are nil
func reduceError(errs []error) []error {
	for i := range errs {
		if errs[i] != nil {
			return errs
		}
	}
	return nil
}

//
func savePredictor(sp Predictor, filename string) error {
	fmt.Println("save file name = ", filename)
	f, err := os.Create(filename)
	if err != nil {
		return err
	}

	jsonBytes, err := json.MarshalIndent(sp, "", "\t")
	if err != nil {
		return err
	}
	f.Write(jsonBytes)
	f.Close()
	return nil
}

func PredictorDirectory(savepath string) string {
	return filepath.Join(savepath, "algorithm")
}

func PredictorFilename(savepath string) string {
	return filepath.Join(savepath, "algorithm", "trained_algorithm.json")
}

func LoadScalePredictor(savepath string) (ScalePredictor, error) {
	p := ScalePredictor{}
	filename := PredictorFilename(savepath)

	f, err := os.Open(filename)
	if err != nil {
		return p, err
	}
	defer f.Close()

	decoder := json.NewDecoder(f)

	decoder.Decode(&p)
	return p, nil
}
