package dataloader

import (
	"fmt"
	"math"
	"sync"

	"github.com/gonum/floats"
)

var identityFunc func([]float64) (float64, error) = func(d []float64) (float64, error) {
	if len(d) != 1 {
		return math.NaN(), fmt.Errorf("Length of data is not 1")
	}
	return d[0], nil
}

// divides the second input by the first
var nondimensionalizer = func(d []float64) (float64, error) {
	if len(d) != 2 {
		return math.NaN(), fmt.Errorf("Length of data is not 2")
	}
	return d[1] / d[0], nil
}

var logFunc func([]float64) (float64, error) = func(d []float64) (float64, error) {
	if len(d) != 1 {
		return math.NaN(), fmt.Errorf("Length of data is not 1")
	}
	return math.Log(d[0]), nil
}

func newSumFunc(neededLength int) func([]float64) (float64, error) {
	return func(d []float64) (float64, error) {
		if len(d) != neededLength {
			return math.NaN(), fmt.Errorf("Length of data does not match")
		}
		return floats.Sum(d), nil
	}
}

// Package for loading data for ransuq

type Dataset struct {
	Name     string // Identifier for the datatype
	Filename string // Path to the data file (includes filename)
	Format   Format // The format of tha dataset
}

type Format interface {
	Fieldmap(string) *FieldTransformer
	ReadFields(fields []string, filename string) ([][]float64, error)
}

type AppendableFormat interface {
	NewAppendFields(oldFilename, newFilename string, newVarnames []string, data [][]float64) error
}

// FieldTransformer
type FieldTransformer struct {
	InternalNames []string                         // Says which fieldnames need to be loaded
	Transformer   func([]float64) (float64, error) // Transforms the fields to the data
}

func Load(fields []string, datasets []*Dataset) ([][][]float64, error) {
	data := make([][][]float64, len(datasets))
	errors := make([]error, len(datasets))
	w := sync.WaitGroup{}
	for i, dataset := range datasets {
		w.Add(1)
		go func(i int, dataset *Dataset) {
			data[i], errors[i] = LoadFromDataset(fields, dataset)
			if errors[i] != nil {
				fmt.Println(dataset.Name, ": error loading: ", errors[i])
			} else {
				fmt.Println(dataset.Name, ": loaded successfully")
			}
			w.Done()
		}(i, dataset)
	}
	w.Wait()
	str := ""
	for i, err := range errors {
		if err != nil {
			str += "dataset " + datasets[i].Name + ": err: " + err.Error()
		}
	}
	if str != "" {
		return nil, fmt.Errorf(str)
	}
	return data, nil
}

// LoadFromDataset loads all of the necessary fields from one dataset
func LoadFromDataset(fields []string, dataset *Dataset) ([][]float64, error) {

	// TODO: Change this to use Matrix

	// First, find which columns are needed for each field
	transformers := make([]*FieldTransformer, len(fields))
	for i, field := range fields {
		transformers[i] = dataset.Format.Fieldmap(field)
		if transformers[i] == nil {
			return nil, fmt.Errorf("Unkown field %v for dataset %v", field, dataset.Name)
		}
	}

	// Next, find all the unique fields
	m := make(map[string]bool)
	for _, transformer := range transformers {
		for _, name := range transformer.InternalNames {
			m[name] = true
		}
	}
	// Next, transform the unique fields into a slice
	nameToCol := make(map[string]int)
	fieldsToRead := make([]string, len(m))
	i := 0
	for key := range m {
		fieldsToRead[i] = key
		nameToCol[key] = i
		i++
	}

	// Read the fields now that they are in the format nomenclature
	fullData, err := dataset.Format.ReadFields(fieldsToRead, dataset.Filename)
	if err != nil {
		return nil, err
	}

	// Create final data structure
	data := make([][]float64, len(fullData))
	for i := range data {
		data[i] = make([]float64, len(fields))
	}

	// Transform the final fields
	for j, transformer := range transformers {
		nFields := len(transformer.InternalNames)
		tmpData := make([]float64, nFields)
		cols := make([]int, nFields)
		for i := range cols {
			cols[i] = nameToCol[transformer.InternalNames[i]]
		}
		for i := range data {
			for k := range tmpData {
				tmpData[k] = fullData[i][cols[k]]
				newValue, err := transformer.Transformer(tmpData)
				if err != nil {
					return nil, err
				}
				data[i][j] = newValue
			}
		}
	}
	return data, nil
}
