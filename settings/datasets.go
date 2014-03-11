package settings

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"github.com/btracey/su2tools/driver"
	"github.com/btracey/su2tools/nondimensionalize"

	"ransuq"
	"ransuq/datawrapper"
)

var gopath string
var supath string

func init() {
	gopath = os.Getenv("GOPATH")
	if gopath == "" {
		panic("gopath not set")
	}

	supath = os.Getenv("SU2_RUN")
	if supath == "" {
		panic("SU2_RUN not set")
	}
}

//
func init() {
	sortedDatasets = append(sortedDatasets, SingleFlatplate)
	sortedDatasets = append(sortedDatasets, NoDataset)

	sort.Strings(sortedDatasets)
}

var sortedDatasets []string

const (
	SingleFlatplate = "single_flatplate"
	MultiFlatplate  = "multi_flatplate"
	FlatplateSweep  = "flatplate_sweep"
	NoDataset       = "none"
)

func GetDatasets(data string) ([]ransuq.Dataset, error) {
	switch data {
	default:
		return nil, Missing{
			Prefix:  "dataset setting not found",
			Options: sortedDatasets,
		}
	case "none":
		return []ransuq.Dataset{}, nil
	case SingleFlatplate:
		dataset := newFlatplate(5e6, "med")
		return []ransuq.Dataset{dataset}, nil
	case MultiFlatplate:
		return []ransuq.Dataset{newFlatplate(3e6, "med"), newFlatplate(5e6, "med"), newFlatplate(7e6, "med")}, nil
	case FlatplateSweep:
		return []ransuq.Dataset{newFlatplate(3e6, "med"), newFlatplate(4e6, "med"), newFlatplate(5e6, "med"), newFlatplate(6e6, "med"), newFlatplate(7e6, "med")}, nil
	}
}

type FlatplateDataset struct {
	datawrapper.SU2
}

func newFlatplate(re float64, fidelity string) ransuq.Dataset {
	basepath := filepath.Join(gopath, "data", "ransuq", "flatplate")
	baseconfig := filepath.Join(basepath, "base_flatplate_config.cfg")

	restring := strconv.FormatFloat(re, 'g', -1, 64)

	name := "Flatplate_Re_" + restring

	// Change the + in the exponent to an underscore
	b := []byte(name)
	for i := range b {
		if b[i] == '+' {
			b[i] = '_'
		}
	}

	name = string(b)

	wd := filepath.Join(basepath, fidelity, name)

	// Create the working directory for writing if it does not exist
	err := os.MkdirAll(wd, 0700)
	if err != nil {
		panic(err)
	}

	// Create the driver
	drive := &driver.Driver{
		Name:      name,
		Config:    "config.cfg",
		Wd:        wd,
		FancyName: "Re " + restring,
		Stdout:    name + "_log.txt",
	}

	// Set the base config options to be those
	err = drive.SetRelativeOptions(baseconfig, false, nil)
	if err != nil {
		panic(err)
	}

	// set mesh file to be the base mesh file
	drive.Options.MeshFilename = filepath.Join(basepath, drive.Options.MeshFilename)

	// set other things
	drive.Options.ReynoldsNumber = re

	drive.Options.RefTemperature = drive.Options.FreestreamTemperature
	//get the freestream pressure and density
	pressure, density := nondimensionalize.Values(drive.Options.FreestreamTemperature, drive.Options.ReynoldsNumber, drive.Options.MachNumber, drive.Options.GasConstant, drive.Options.ReynoldsLength, drive.Options.GammaValue)
	drive.Options.RefPressure = pressure
	drive.Options.RefDensity = density
	totalT := nondimensionalize.TotalTemperature(drive.Options.FreestreamTemperature, drive.Options.MachNumber, drive.Options.GammaValue)
	totalP := nondimensionalize.TotalPressure(pressure, drive.Options.MachNumber, drive.Options.GammaValue)
	totalTString := strconv.FormatFloat(totalT, 'g', 16, 64)
	totalPString := strconv.FormatFloat(totalP, 'g', 16, 64)
	pString := strconv.FormatFloat(pressure, 'g', 16, 64)
	drive.Options.MarkerInlet = "( inlet, " + totalTString + ", " + totalPString + ", 1.0, 0.0, 0.0 )"
	drive.Options.MarkerOutlet = "( outlet, " + pString + ", farfield, " + pString + " )"

	switch fidelity {
	case "low":
		drive.Options.ResidualReduction = 0.2
	case "med":
		drive.Options.ResidualReduction = 5
	case "high":
		drive.Options.ResidualReduction = 7
	default:
		panic("bad fidelity")
	}

	// Create an SU2 datawrapper from it
	return &datawrapper.SU2{
		Driver:      drive,
		Su2Caller:   driver.Serial{}, // TODO: Need to figure out how to do this better
		IgnoreNames: []string{"YLoc"},
		IgnoreFunc:  func(d []float64) bool { return d[0] < 1e-10 },
		Name:        name,
	}
}
