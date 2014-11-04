package quickload

import (
	"github.com/btracey/ransuq"
	"github.com/btracey/ransuq/settings"
	"github.com/btracey/su2tools/driver"
	"github.com/gonum/matrix/mat64"
)

func Load(dataset, features string) (inputData, outputData *mat64.Dense, weights []float64, err error) {
	datasets, err := settings.GetDatasets(dataset, driver.Serial{})
	if err != nil {
		return nil, nil, nil, err
	}

	inputFeatures, outputFeatures, err := settings.GetFeatures(features)
	if err != nil {
		return nil, nil, nil, err
	}

	inputDataMat, outputDataMat, weights, err := ransuq.DenseLoadAll(datasets, inputFeatures, outputFeatures, nil, nil)
	if err != nil {
		return nil, nil, nil, err
	}

	return inputDataMat.(*mat64.Dense), outputDataMat.(*mat64.Dense), weights, nil
}

func LoadExtra(dataset string, features []string) (data *mat64.Dense, weights []float64, err error) {
	datasets, err := settings.GetDatasets(dataset, driver.Serial{})
	if err != nil {
		return nil, nil, err
	}
	dataMat, _, weights, err := ransuq.DenseLoadAll(datasets, features, []string{}, nil, nil)
	return dataMat.(*mat64.Dense), weights, err
}
