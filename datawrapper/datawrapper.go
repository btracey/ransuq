// package datawrapper is a set of wrappers for loading data from various styles

package datawrapper

import (
	"fmt"
	"os"
	"path/filepath"
	"ransuq"
	"ransuq/dataloader"

	"github.com/btracey/su2tools/config"
	"github.com/btracey/su2tools/driver"
	"github.com/reggo/reggo/common"

	"github.com/gonum/matrix/mat64"
)

// SU2 is a type for loading SU2 data and running SU2 Cases
type SU2 struct {
	Driver      *driver.Driver
	Su2Caller   driver.SU2Syscaller
	IgnoreNames []string
	IgnoreFunc  func([]float64) bool
	Name        string
}

func (su *SU2) ID() string {
	return su.Name
}

func (su *SU2) Identifier() string {
	return su.Name
}

func (su *SU2) SetSyscaller(sys driver.SU2Syscaller) {
	su.Su2Caller = sys
}

func (su *SU2) Load(fields []string) (common.RowMatrix, error) {
	// Construct a dataloader
	loader := &dataloader.Dataset{
		Name:     su.Driver.Name,
		Filename: su.Driver.Fullpath(su.Driver.Options.SolutionFlowFilename),
		Format:   &dataloader.SU2_restart_2dturb{},
	}

	// Load the needed fields from the data
	tmpData, err := dataloader.LoadFromDataset(fields, loader)
	if err != nil {
		return nil, err
	}

	// Load the fields needed to find ingore data
	ignoreData, err := dataloader.LoadFromDataset(su.IgnoreNames, loader)

	nSamples := len(tmpData)
	nDim := len(tmpData[0])

	data := mat64.NewDense(nSamples, nDim, nil) // Allocate memory for enough samples

	var nRows int
	for i := range tmpData {
		if su.IgnoreFunc(ignoreData[i]) {
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

func (su *SU2) Generated() bool {
	_ = ransuq.Comparable(su)
	fmt.Println("In ", su.Name, " generated")
	b := su.Driver.IsComputed()
	fmt.Println(su.Name, "computed?: ", b)
	return b
}

func (su *SU2) Run() error {
	fmt.Println("Started running ", su.Driver.Name)
	err := su.Driver.Run(su.Su2Caller)
	if err != nil {
		return err
	}
	return su.Driver.CopyRestartToSolution()
}

func (su *SU2) Comparison(algfile string, outLoc string, featureSet string) (ransuq.Generatable, error) {
	// Copy the config file and run it.
	newOptionList := make(config.OptionList)
	drive := su.Driver

	for key, val := range drive.OptionList {
		newOptionList[key] = val
	}

	newName := drive.Name + "_ml"
	newDir := filepath.Join(outLoc, newName)
	err := os.MkdirAll(newDir, 0700)
	if err != nil {
		return nil, err
	}
	mlDriver := &driver.Driver{
		Name:       newName,
		Options:    drive.Options.Copy(),
		Config:     drive.Config,
		Wd:         filepath.Join(newDir, "su2run"),
		Stdout:     newName + "_log.txt",
		OptionList: newOptionList,
		FancyName:  drive.FancyName + " ML",
	}

	// Edit the options
	// Need to not care if the options have relative or not paths
	// use filepath.IsAbs
	if !filepath.IsAbs(mlDriver.Options.MeshFilename) {
		mlDriver.Options.MeshFilename = mlDriver.Fullpath(mlDriver.Options.MeshFilename)
	}
	// Now, change the turbulence model to SA, and add the json file
	mlDriver.Options.KindTurbModel = "ML"
	mlDriver.Options.MlTurbModelFile = algfile
	mlDriver.OptionList["MlTurbModelFile"] = true
	mlDriver.Options.ExtIter = 9999.0
	mlDriver.Options.MlTurbModelFeatureset = featureSet

	newSu2 := &SU2ML{
		SU2: &SU2{
			Driver:      mlDriver,
			Su2Caller:   su.Su2Caller,
			IgnoreNames: su.IgnoreNames,
			IgnoreFunc:  su.IgnoreFunc,
			Name:        newName,
		},
		OrigDriver: su.Driver,
		SaveDir:    newDir,
	}
	return newSu2, nil
}

type SU2ML struct {
	*SU2
	OrigDriver *driver.Driver
	SaveDir    string
}

func (su *SU2ML) PostProcess() error {
	fmt.Println("In post-process for ML Case")

	_ = ransuq.PostProcessor(su)

	// Want to call the flat-plate post process
	path := filepath.Join(su.SaveDir, "postprocess")
	fmt.Println("path is for post process save", path)
	err := flatplateCompare([]*driver.Driver{su.OrigDriver, su.SU2.Driver}, path)
	if err != nil {
		panic(err)
	}
	fmt.Println("done post-process")

	return nil
}
