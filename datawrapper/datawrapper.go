// package datawrapper is a set of wrappers for loading data from various styles

package datawrapper

import (
	"encoding/csv"
	"errors"
	"fmt"
	"image/color"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"

	"github.com/btracey/ransuq"
	"github.com/btracey/ransuq/dataloader"
	"github.com/btracey/su2tools/config/enum"
	"github.com/btracey/su2tools/driver"

	"github.com/reggo/reggo/common"

	"github.com/gonum/matrix/mat64"
)

// SU2 is a type for loading SU2 data and running SU2 Cases
type SU2 struct {
	Driver                  *driver.Driver
	Su2Caller               driver.Syscaller
	IgnoreNames             []string
	IgnoreFunc              func([]float64) bool
	Name                    string
	ComparisonPostprocessor Postprocessor
	ExtraMlStrings          []string
	ComparisonNameAddendum  string // Additional string to append after _ML in the comparison file
}

func (su *SU2) ID() string {
	return su.Name
}

func (su *SU2) NumCores() int {
	return su.Su2Caller.NumCores()
}

func (su *SU2) Identifier() string {
	return su.Name
}

func (su *SU2) SetSyscaller(sys driver.Syscaller) {
	su.Su2Caller = sys
}

func (su *SU2) Load(fields []string) (common.RowMatrix, error) {
	/*
		reflength := su.ReferenceLength
		if reflength == 0 {
			reflength = 1
		}
	*/

	// Construct a dataloader
	loader := &dataloader.Dataset{
		Name:     su.Driver.Name,
		Filename: filepath.Join(su.Driver.Wd, su.Driver.Options.SolutionFlowFilename),
		Format:   &dataloader.SU2_restart_2dturb{},
	}

	// Load the needed fields from the data
	tmpData, err := dataloader.LoadFromDataset(fields, loader)
	if err != nil {
		return nil, err
	}

	// Load the fields needed to find ingore data
	ignoreData, err := dataloader.LoadFromDataset(su.IgnoreNames, loader)

	if err != nil {
		return nil, err
	}

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
	data = (data.View(0, 0, nRows, nDim)).(*mat64.Dense)
	return data, nil
}

func (su *SU2) Generated() bool {
	_ = ransuq.Comparable(su)

	status := su.Driver.Status()

	b := su.Driver.IsComputed(status)
	if !b {
		fmt.Println("Not computed: ", b)
	}
	return b

	/*
		status := su.Driver.Status()
		if status == driver.Computed {
			return true
		}
	*/

	/*
		fmt.Println("Not computed because: ", status)

		if status == driver.UnequalOptions {
			f := filepath.Join(su.Driver.Wd, su.Driver.Config)
			otherOptions, _, err := config.ReadFromFile(f)
			if err != nil {
				panic(err)
			}

			fmt.Println(config.Diff(otherOptions, su.Driver.Options))
		}
	*/
}

func (su *SU2) Run() error {
	err := su.Driver.Run(su.Su2Caller)
	if err != nil {
		return err
	}
	return su.Driver.CopyRestartToSolution()
}

func (su *SU2) Comparison(algfile string, outLoc string, featureSet string) (ransuq.Generatable, error) {
	// Copy the config file and run it.
	//newOptionList := make(config.OptionList)
	drive := su.Driver

	/*
		for key, val := range drive.OptionList {
			newOptionList[key] = val
		}
	*/

	newName := drive.Name + "_ml" + su.ComparisonNameAddendum
	newDir := filepath.Join(outLoc, newName)
	postprocessDir := filepath.Join(newDir, "postprocess")
	err := os.MkdirAll(newDir, 0700)
	if err != nil {
		return nil, err
	}

	wd := filepath.Join(newDir, "su2run")

	fmt.Println("wd is ", wd)
	mlDriver := &driver.Driver{
		Name:    newName,
		Options: drive.Options.Copy(),
		Config:  drive.Config,
		Wd:      wd,
		Stdout:  newName + "_log.txt",
		//OptionList: newOptionList,
		OptionList: nil,
		FancyName:  drive.FancyName + " ML " + su.ComparisonNameAddendum,
	}

	// Edit the options
	// Need to not care if the options have relative or not paths
	// use filepath.IsAbs

	// First, get the absolute path of the mesh name in the mlDrive
	absMesh, err := filepath.Abs(filepath.Join(drive.Wd, mlDriver.Options.MeshFilename))
	if err != nil {
		return nil, err
	}

	fmt.Println("first abs mesh is", absMesh)

	// Get the relative mesh path
	relMesh, err := filepath.Rel(wd, absMesh)
	if err != nil {
		return nil, err
	}

	fmt.Println("algfile is ", algfile)

	// Alg filename given by me so it's an absolute path
	relAlgFile, err := filepath.Rel(wd, algfile)
	if err != nil {
		return nil, err
	}
	fmt.Println("rel algfile is", relAlgFile)
	fmt.Println("rel mesh is", relMesh)

	// Now, change the turbulence model to SA, and add the json file
	mlDriver.Options.KindTurbModel = enum.Ml
	mlDriver.Options.MeshFilename = relMesh
	mlDriver.Options.MlTurbModelFile = relAlgFile
	mlDriver.Options.ExtraOutput = true
	//mlDriver.OptionList["MlTurbModelFile"] = true
	//mlDriver.OptionList["ExtraOutput"] = true
	mlDriver.Options.ExtIter = 9999.0
	mlDriver.Options.MlTurbModelFeatureset = featureSet

	// Add in the extra strings (say, for just doing ML in BL)
	extraString := make([]string, len(su.ExtraMlStrings))
	copy(extraString, su.ExtraMlStrings)
	mlDriver.Options.MlTurbModelExtra = extraString

	newSu2 := &SU2ML{
		SU2: &SU2{
			Driver:      mlDriver,
			Su2Caller:   su.Su2Caller,
			IgnoreNames: su.IgnoreNames,
			IgnoreFunc:  su.IgnoreFunc,
			Name:        newName,
		},
		OrigDriver:              su.Driver,
		PostprocessDir:          postprocessDir,
		ComparisonPostprocessor: su.ComparisonPostprocessor,
	}
	return newSu2, nil
}

type SU2ML struct {
	*SU2
	OrigDriver              *driver.Driver
	PostprocessDir          string
	ComparisonPostprocessor Postprocessor
}

func (su *SU2ML) PostProcess() error {

	status := su.Driver.Status()
	if status != driver.ComputedSuccessfully {
		if status == driver.ComputedWithError {
			return nil
		}
		if status != driver.ComputedSuccessfully {
			return errors.New(status.String())
		}
	}

	// Maybe the comparison needs to happen outside in the RANS uq function
	// so that it has the dataloader stuff out there (don't bring it in here)

	/*
		// Make a plot for the prediction vs. truth at the final state
		comparisonDir := filepath.Join(su.PostprocessDir, "comparison")

		err := os.Mkdir(comparisonDir, 0700)
		if err != nil {
			return errors.New("error making ml postprocess comparison directory: ", err.Error())
		}
	*/

	// Need to collect the prediction and true values. Some how this needs to come
	// from the dataloader. So far it can

	if su.ComparisonPostprocessor != nil {
		return su.ComparisonPostprocessor.PostProcess(su)
	}
	return nil
}

type Postprocessor interface {
	PostProcess(*SU2ML) error
}

type FlatplatePostprocessor struct {
}

func (FlatplatePostprocessor) PostProcess(su *SU2ML) error {
	// Want to call the flat-plate post process
	//path := filepath.Join(su.SaveDir, "postprocess")
	path := su.PostprocessDir
	fmt.Println("path is for post process save", path)
	err := flatplateCompare([]*driver.Driver{su.OrigDriver, su.SU2.Driver}, path)
	if err != nil {
		panic(err)
	}
	fmt.Println("done post-process")

	return nil
}

type AirfoilPostprocessor struct{}

func (AirfoilPostprocessor) PostProcess(su *SU2ML) error {
	// Make a Cf plot comparing the ml and the original
	err := makeCfPlot(su.OrigDriver, su.SU2.Driver, su.PostprocessDir)
	if err != nil {
		panic(err)
	}
	fmt.Println("done post-process")
	return err
}

func makeCfPlot(orig, ml *driver.Driver, postprocessdir string) error {
	// Load the surface files for the drivers
	err := os.MkdirAll(postprocessdir, 0700)
	if err != nil {
		return err
	}
	filename := filepath.Join(postprocessdir, "cfplot.pdf")
	// See if the file exists
	if _, err = os.Stat(filename); err == nil {
		return nil
	}

	origSurfFilename := filepath.Join(orig.Wd, orig.Options.SurfaceFlowFilename)
	newSurfFilename := filepath.Join(ml.Wd, ml.Options.SurfaceFlowFilename)

	// hardcode in the csv suffix because it's added automatically by SU2
	origSurfFilename += ".csv"
	newSurfFilename += ".csv"

	origSurf, err := os.Open(origSurfFilename)
	defer origSurf.Close()
	if err != nil {
		return err
	}

	newSurf, err := os.Open(newSurfFilename)
	defer newSurf.Close()
	if err != nil {
		return err
	}

	oldCfs, err := readCfInfo(origSurf)
	if err != nil {
		panic(err)
	}
	newCfs, err := readCfInfo(newSurf)
	if err != nil {
		panic(err)
	}

	return plotCfs(oldCfs, newCfs, filename)
}

func plotCfs(oldCfs, newCfs []float64, filename string) error {
	// Get the points
	oldPts := make(plotter.XYs, len(oldCfs))
	newPts := make(plotter.XYs, len(newCfs))
	for i := range oldCfs {
		oldPts[i].X = float64(i) / float64(len(oldCfs))
		oldPts[i].Y = oldCfs[i]
		newPts[i].X = float64(i) / float64(len(oldCfs))
		newPts[i].Y = newCfs[i]
	}

	p, err := plot.New()
	if err != nil {
		return err
	}

	/*
		fmt.Println("old pts = ", oldPts)
		fmt.Println(newPts)
	*/
	p.X.Min = 0
	p.X.Max = 1
	p.X.Label.Text = "x/c"
	p.Y.Label.Text = "Cf"
	ticks := []plot.Tick{
		{Value: 0, Label: "0"},
		{Value: 0.5, Label: "0.5"},
		{Value: 1, Label: "1"},
	}
	p.X.Tick.Marker = plot.ConstantTicks(ticks)

	trueScatter, err := plotter.NewScatter(oldPts)
	if err != nil {
		return err
	}
	mlScatter, err := plotter.NewScatter(newPts)
	if err != nil {
		return err
	}
	trueScatter.GlyphStyle.Color = color.RGBA{R: 255}
	mlScatter.GlyphStyle.Radius = 2

	mlScatter.GlyphStyle.Color = color.RGBA{B: 255}
	mlScatter.GlyphStyle.Radius = 2

	p.Add(trueScatter)
	p.Add(mlScatter)
	p.Legend.Add("True", trueScatter)
	p.Legend.Add("ML", mlScatter)
	p.Legend.Top = true
	p.Legend.Left = true

	if err := p.Save(4, 4, filename); err != nil {
		return err
	}
	return nil
}

func readCfInfo(r io.Reader) ([]float64, error) {
	cr := csv.NewReader(r)
	cr.LazyQuotes = true
	records, err := cr.ReadAll()
	if err != nil {
		fmt.Println("error reading all")
		panic(err)
		return nil, err
	}
	// Search the first record for the skin friction coefficient
	idx := -1
	for i, s := range records[0] {
		s = strings.TrimSpace(s)
		s = strings.TrimPrefix(s, "\"")
		s = strings.TrimSuffix(s, "\"")
		if s == "Skin_Friction_Coefficient" {
			idx = i
			break
		}
	}
	if idx == -1 {
		return nil, errors.New("cf not found")
	}
	// Now, extract all of them in order
	cfs := make([]float64, len(records[1:]))
	for i, v := range records[1:] {
		s := v[idx]
		s = strings.TrimSpace(s)
		cfs[i], err = strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, err
		}
	}
	return cfs, nil
}
