package settings

import (
	"errors"
	"path/filepath"

	"github.com/btracey/ransuq"
	"github.com/btracey/ransuq/datawrapper"
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
	caller driver.Syscaller,
	extraStringsSetting []string,
) (*ransuq.Settings, error) {
	// Get the training data sets
	trainingData, err := GetDatasets(training, caller)
	if err != nil {
		return nil, errors.New("training " + err.Error())
	}

	baseTestingData, err := GetDatasets(testing, caller)
	if err != nil {
		return nil, errors.New("testing " + err.Error())
	}

	var testingData []ransuq.Dataset

	for i, set := range extraStringsSetting {
		extraStrings, err := GetSU2ExtraStrings(set)
		if err != nil {
			return nil, errors.New("error getting extra strings: " + err.Error())
		}
		// For the testing data, loop over the datasets to see if they are SU2 sets.
		// If so, pass the extra strings on
		for _, dataset := range baseTestingData {
			su2, ok := dataset.(*datawrapper.SU2)
			if ok {
				// Need to copy the dataset. Hopefully these pointers don't mess with anything...
				newSU2 := &datawrapper.SU2{
					Driver:      su2.Driver,
					Su2Caller:   su2.Su2Caller,
					IgnoreNames: su2.IgnoreNames,
					IgnoreFunc:  su2.IgnoreFunc,
					Name:        su2.Name,
					ComparisonPostprocessor: su2.ComparisonPostprocessor,
					ExtraMlStrings:          extraStrings,
					ComparisonNameAddendum:  set,
				}

				testingData = append(testingData, newSU2)
			} else {
				// Since it's not an SU2, nothing changes, so only add it once
				if i == 0 {
					testingData = append(testingData, dataset)
				}
			}

		}
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
