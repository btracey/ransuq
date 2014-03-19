package settings

import (
	"errors"
	"path/filepath"

	"github.com/btracey/ransuq"
	"github.com/btracey/su2tools/driver"
)

// GetSettings returns a populated settings structure for the given options
func GetSettings(
	training,
	testing,
	features,
	weightSet,
	algorithm,
	trainSettings string,
	caller driver.SU2Syscaller,
) (*ransuq.Settings, error) {
	// Get the training data sets
	trainingData, err := GetDatasets(training, caller)
	if err != nil {
		return nil, errors.New("training " + err.Error())
	}

	testingData, err := GetDatasets(testing, caller)
	if err != nil {
		return nil, errors.New("testing " + err.Error())
	}

	// Get the input and output features
	inputs, outputs, err := GetFeatures(features)
	if err != nil {
		return nil, err
	}

	// Get the weights
	weights, f, err := GetWeight(weightSet)
	if err != nil {
		return nil, err
	}

	// Get the algorithm
	trainer, err := GetTrainer(trainSettings, algorithm, len(inputs), len(outputs))
	if err != nil {
		return nil, err
	}

	set := &ransuq.Settings{
		TrainingData:   trainingData,
		TestingData:    testingData,
		FeatureSet:     features,
		InputFeatures:  inputs,
		OutputFeatures: outputs,
		WeightFeatures: weights,
		WeightFunc:     f,
		Savepath:       filepath.Join(gopath, "results", "ransuq", features, weightSet, algorithm, trainSettings, training),
		Trainer:        trainer,
	}
	return set, nil
}
