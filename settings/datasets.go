package settings

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"github.com/btracey/su2tools/config/enum"
	"github.com/btracey/su2tools/config/su2types"
	"github.com/btracey/su2tools/driver"
	"github.com/btracey/su2tools/nondimensionalize"

	"github.com/btracey/ransuq"
	"github.com/btracey/ransuq/datawrapper"
	"github.com/btracey/ransuq/synthetic"
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
	sortedDatasets = append(sortedDatasets,
		NoDataset,
		SingleFlatplate,
		MultiFlatplate,
		FlatplateSweep,
		SyntheticFlatplateProduction,
		SingleRae,
	)
	sort.Strings(sortedDatasets)
}

var sortedDatasets []string

const (
	NoDataset                    = "none"
	SingleFlatplate              = "single_flatplate"
	MultiFlatplate               = "multi_flatplate"
	MultiFlatplateBL             = "multi_flatplate_bl"
	FlatplateSweep               = "flatplate_sweep"
	SyntheticFlatplateProduction = "synth_flat_prod"
	MultiAndSynthFlatplate       = "multi_and_sync_flatplate"
	SingleRae                    = "single_rae"
	ExtraFlatplate               = "extra_flatplate"
	LES4                         = "les4"
	LES4Tenth                    = "les4_tenth"
	SingleNaca0012               = "single_naca_0012"
	MultiNaca0012                = "multi_naca_0012"
	Naca0012Sweep                = "naca_0012_sweep"
)

// All of these assume that the working directory is $GOPATH, which should be set
// from the main script

func GetDatasets(data string, caller driver.Syscaller) ([]ransuq.Dataset, error) {
	var datasets []ransuq.Dataset

	flatplate3_06 := newFlatplate(3e6, "med", "atwall")
	flatplate4_06 := newFlatplate(4e6, "med", "atwall")
	flatplate5_06 := newFlatplate(5e6, "med", "atwall")
	flatplate6_06 := newFlatplate(6e6, "med", "atwall")
	flatplate7_06 := newFlatplate(7e6, "med", "atwall")

	flatplate3_06_BL := newFlatplate(3e6, "med", "justbl")
	//flatplate4_06_BL := newFlatplate(4e6, "med", "justbl")
	flatplate5_06_BL := newFlatplate(5e6, "med", "justbl")
	//flatplate6_06_BL := newFlatplate(6e6, "med", "justbl")
	flatplate7_06_BL := newFlatplate(7e6, "med", "justbl")

	flatplateSweep := []ransuq.Dataset{flatplate3_06, flatplate4_06, flatplate5_06, flatplate6_06, flatplate7_06}
	multiFlatplate := []ransuq.Dataset{flatplate3_06, flatplate5_06, flatplate7_06}

	switch data {
	default:
		return nil, Missing{
			Prefix:  "dataset setting not found",
			Options: sortedDatasets,
		}
	case "none":
		datasets = []ransuq.Dataset{}
	case SingleFlatplate:
		datasets = []ransuq.Dataset{flatplate5_06}
	case MultiFlatplate:
		datasets = multiFlatplate
	case MultiFlatplateBL:
		datasets = []ransuq.Dataset{flatplate3_06_BL, flatplate5_06_BL, flatplate7_06_BL}
	case ExtraFlatplate:
		datasets = []ransuq.Dataset{newFlatplate(1e6, "med", "atwall"), newFlatplate(2e6, "med", "atwall"), newFlatplate(1.5e6, "med", "atwall")}
	case FlatplateSweep:
		datasets = flatplateSweep
	case SyntheticFlatplateProduction:
		datasets = []ransuq.Dataset{synthetic.Production{synthetic.FlatplateBounds}}
	case MultiAndSynthFlatplate:
		datasets = []ransuq.Dataset{synthetic.Production{synthetic.FlatplateBounds}}
		datasets = append(datasets, flatplateSweep...)
	case SingleRae:
		datasets = []ransuq.Dataset{newAirfoil()}
	case LES4:
		datasets = []ransuq.Dataset{
			&datawrapper.CSV{
				Location:   filepath.Join(gopath, "data", "ransuq", "LES", "exp4.txt"),
				Name:       "LES_exp4",
				IgnoreFunc: func([]float64) bool { return false },
			},
		}
	case LES4Tenth:
		datasets = []ransuq.Dataset{
			&datawrapper.CSV{
				Location: filepath.Join(gopath, "data", "ransuq", "LES", "exp4_mod.txt"),
				Name:     "LES_exp4",
				IgnoreFunc: func(a []float64) bool {
					intpoint := int(a[0])
					return (intpoint % 10) != 0
				},
				IgnoreNames: []string{"Datapoint"},
			},
		}
	case SingleNaca0012:
		datasets = []ransuq.Dataset{
			newNaca0012(0),
		}
	case MultiNaca0012:
		datasets = []ransuq.Dataset{
			newNaca0012(0),
			newNaca0012(3),
			newNaca0012(6),
			newNaca0012(9),
			newNaca0012(12),
		}
	case Naca0012Sweep:
		datasets = []ransuq.Dataset{
			newNaca0012(0),
			newNaca0012(1),
			newNaca0012(2),
			newNaca0012(3),
			newNaca0012(4),
			newNaca0012(5),
			newNaca0012(6),
			newNaca0012(7),
			newNaca0012(8),
			newNaca0012(9),
			newNaca0012(10),
			newNaca0012(11),
			newNaca0012(12),
		}
	}

	for _, dataset := range datasets {
		su2, ok := dataset.(*datawrapper.SU2)
		if ok {
			fmt.Println("in setting syscaller")
			su2.SetSyscaller(caller)
		}
	}
	return datasets, nil
}

type FlatplateDataset struct {
	datawrapper.SU2
}

func newFlatplate(re float64, fidelity string, ignoreType string) ransuq.Dataset {
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

	baseconfigFile, err := os.Open(baseconfig)
	if err != nil {
		return nil
	}

	// Load in the existing
	err = drive.Load(baseconfigFile)
	if err != nil {
		panic(err)
	}

	// set mesh file to be the base mesh file but use relative path

	fullMeshFilename := filepath.Join(basepath, drive.Options.MeshFilename)

	relMeshFilename, err := filepath.Rel(wd, fullMeshFilename)
	if err != nil {
		panic(err)
	}
	drive.Options.MeshFilename = relMeshFilename

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
	//pString := strconv.FormatFloat(pressure, 'g', 16, 64)
	//drive.Options.MarkerInlet = &su2types.Inlet{"( inlet, " + totalTString + ", " + totalPString + ", 1.0, 0.0, 0.0 )"}
	drive.Options.MarkerInlet = &su2types.Inlet{Strings: []string{"inlet", totalTString, totalPString, "1", "0", "0"}}
	drive.Options.MarkerOutlet = &su2types.StringDoubleList{
		Strings: []string{"outlet", "farfield"},
		Doubles: []float64{pressure, pressure},
	}
	//drive.Options.MarkerOutlet = &su2types.StringDoubleList{"( outlet, " + pString + ", farfield, " + pString + " )"}

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

	var ignoreFunc func(d []float64) bool
	var ignoreNames []string
	switch ignoreType {
	case "atwall":
		ignoreNames = []string{"YLoc"}
		ignoreFunc = func(d []float64) bool { return d[0] < 1e-10 }
	case "justbl":
		ignoreNames = []string{"YLoc", "XLoc"}
		ignoreFunc = func(d []float64) bool {
			xloc := d[1]
			yloc := d[0]
			if yloc < 1e-10 {
				return true
			}
			if xloc < 0 {
				return true
			}
			if yloc > 0.06 {
				return true
			}
			return false
		}
	default:
		panic("unknown ignore type")
	}

	// Create an SU2 datawrapper from it
	return &datawrapper.SU2{
		Driver:      drive,
		Su2Caller:   driver.Serial{}, // TODO: Need to figure out how to do this better
		IgnoreNames: ignoreNames,
		IgnoreFunc:  ignoreFunc,
		Name:        name,
		ComparisonPostprocessor: datawrapper.FlatplatePostprocessor{},
	}
}

func newAirfoil() ransuq.Dataset {
	basepath := filepath.Join(gopath, "data", "ransuq", "airfoil", "rae")
	configName := "turb_SA_RAE2822.cfg"
	meshName := "mesh_RAE2822_turb.su2"
	baseconfig := filepath.Join(basepath, "testcase", configName)
	meshFile := filepath.Join(basepath, "testcase", meshName)

	fmt.Println("base config is", baseconfig)

	name := "Rae_Base"

	wd := filepath.Join(basepath, "RAE")

	// Create the driver
	drive := &driver.Driver{
		Name:      name,
		Config:    configName,
		Wd:        wd,
		FancyName: "RAE Base",
		Stdout:    name + "_log.txt",
	}

	baseconfigFile, err := os.Open(baseconfig)
	if err != nil {
		panic(err)
	}

	// Set the base config options to be those
	err = drive.Load(baseconfigFile)
	if err != nil {
		panic(err)
	}

	// set mesh file to be the base mesh file
	relMeshName, err := filepath.Rel(wd, meshFile)
	if err != nil {
		panic(err)
	}
	drive.Options.MeshFilename = relMeshName
	drive.Options.KindTurbModel = enum.Ml
	drive.Options.MlTurbModelFile = "none"
	drive.Options.MlTurbModelFeatureset = "SA"
	drive.Options.ExtraOutput = true
	drive.OptionList["MlTurbModelFile"] = true
	drive.OptionList["MlTurbModelFeatureset"] = true
	drive.OptionList["ExtraOutput"] = true

	// Create an SU2 datawrapper from it
	return &datawrapper.SU2{
		Driver:      drive,
		Su2Caller:   driver.Serial{}, // TODO: Need to figure out how to do this better
		IgnoreNames: []string{"YLoc"},
		IgnoreFunc:  func(d []float64) bool { return d[0] < 1e-10 },
		Name:        name,
	}
}

func newNaca0012(aoa float64) ransuq.Dataset {
	basepath := filepath.Join(gopath, "data", "ransuq", "airfoil", "naca0012")
	configName := "naca0012.cfg"
	meshName := "mesh_NACA0012_turb_897x257.su2"
	baseconfig := filepath.Join(basepath, "ransuqbase", configName)
	meshFile := filepath.Join(basepath, "ransuqbase", meshName)

	aoaString := strconv.FormatFloat(aoa, 'g', 16, 64)

	name := "Naca0012_" + aoaString

	wd := filepath.Join(basepath, "Naca0012_"+aoaString)

	// Create the driver
	drive := &driver.Driver{
		Name:      name,
		Config:    configName,
		Wd:        wd,
		FancyName: "NACA 0012 AOA = " + aoaString,
		Stdout:    name + "_log.txt",
	}

	baseconfigFile, err := os.Open(baseconfig)
	if err != nil {
		panic(err)
	}

	// Set the base config options to be those
	err = drive.Load(baseconfigFile)
	if err != nil {
		panic(err)
	}

	drive.Options.Aoa = aoa

	// set mesh file to be the base mesh file
	relMeshName, err := filepath.Rel(wd, meshFile)
	if err != nil {
		panic(err)
	}
	drive.Options.MeshFilename = relMeshName
	drive.Options.KindTurbModel = enum.Ml
	drive.Options.MlTurbModelFile = "none"
	drive.Options.MlTurbModelFeatureset = "SA"
	drive.Options.ExtraOutput = true
	drive.OptionList["MlTurbModelFile"] = true
	drive.OptionList["MlTurbModelFeatureset"] = true
	drive.OptionList["ExtraOutput"] = true

	// Create an SU2 datawrapper from it
	return &datawrapper.SU2{
		Driver:      drive,
		Su2Caller:   driver.Serial{}, // TODO: Need to figure out how to do this better
		IgnoreNames: []string{"YLoc"},
		IgnoreFunc:  func(d []float64) bool { return d[0] < 1e-10 },
		Name:        name,
	}
}

/*
// LES Dataset for Karthik
type LesKarthik struct {
	Int         int

}



	tmpData, err := dataloader.LoadFromDataset(fields, loader)
	if err != nil {
		return nil, err
	}
}

func getLesKarthik(names []string) []ransuq.Dataset {

}
*/
