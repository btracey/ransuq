package datawrapper

import (
	"github.com/btracey/ransuq/dataloader"
	"github.com/gonum/matrix/mat64"
	"github.com/reggo/reggo/common"
)

// CSV is a wrapper aronud CSV files
type CSV struct {
	Location    string
	Name        string
	IgnoreNames []string
	IgnoreFunc  func([]float64) bool
	FieldMap    map[string]string
}

func (csv *CSV) ID() string {
	return csv.Name
}

func (csv *CSV) Load(fields []string) (common.RowMatrix, error) {
	loader := &dataloader.Dataset{
		Name:     csv.Name,
		Filename: csv.Location,
		Format: &dataloader.NaiveCSV{
			FieldMap: csv.FieldMap,
		},
	}
	return loadFromDataloader(fields, loader, csv.IgnoreNames, csv.IgnoreFunc)
}

func loadFromDataloader(fields []string, loader *dataloader.Dataset, ignoreNames []string,
	ignoreFunc func([]float64) bool) (common.RowMatrix, error) {

	// Load the needed fields from the data
	tmpData, err := dataloader.LoadFromDataset(fields, loader)
	if err != nil {
		return nil, err
	}

	// Load the fields needed to find ingore data
	ignoreData, err := dataloader.LoadFromDataset(ignoreNames, loader)
	if err != nil {
		return nil, err
	}

	nSamples := len(tmpData)
	nDim := len(tmpData[0])

	data := mat64.NewDense(nSamples, nDim, nil) // Allocate memory for enough samples

	var nRows int
	for i := range tmpData {
		if ignoreFunc(ignoreData[i]) {
			continue
		}
		for j := 0; j < nDim; j++ {
			data.Set(nRows, j, tmpData[i][j])
		}
		nRows++
	}
	data.View(data, 0, 0, nRows, nDim)
	return data, nil
}
