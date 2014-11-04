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

	lavalLoc = filepath.Join(gopath, "data", "ransuq", "laval", "laval_csv_computed.dat")
}

var sortedDatasets []string

const (
	NoDataset                    = "none"
	NacaPressureFlat             = "naca_pressure_flat"
	NacaPressureFlatBl           = "naca_pressure_flat_bl"
	SingleFlatplate              = "single_flatplate"
	SingleFlatplateBL            = "single_flatplate_bl"
	SingleFlatplateBudget        = "single_flatplate_budget"
	MultiFlatplate               = "multi_flatplate"
	MultiFlatplateBL             = "multi_flatplate_bl"
	MultiFlatplateBudgetBL       = "multi_flatplate_budget_bl"
	FlatplateSweep               = "flatplate_sweep"
	FlatplateSweepBl             = "flatplate_sweep_bl"
	SyntheticFlatplateProduction = "synth_flat_prod"
	MultiAndSynthFlatplate       = "multi_and_sync_flatplate"
	SingleRae                    = "single_rae"
	ExtraFlatplate               = "extra_flatplate"
	LES4                         = "les4"
	LES4Tenth                    = "les4_tenth"
	FwNACA0012                   = "fw_naca0012_shivaji"
	SingleNaca0012               = "single_naca_0012"
	SingleNaca0012Bl             = "single_naca_0012_bl"
	MultiNaca0012                = "multi_naca_0012"
	MultiNaca0012Bl              = "multi_naca_0012_bl"
	Naca0012Sweep                = "naca_0012_sweep"
	PressureGradientMulti        = "pressure_gradient_multi"
	PressureGradientMultiSmall   = "pressure_gradient_multi_small"
	PressureBl                   = "pressure_bl"
	FlatPressureBl               = "flat_pressure_bl"
	DNS5n                        = "dns_5n"
	FlatPress                    = "flat_press"
	LavalDNS                     = "laval_dns"
	LavalDNSCrop                 = "laval_dns_crop"
	LavalDNSBL                   = "laval_dns_bl"
	LavalDNSBLAll                = "laval_dns_bl_all"
	ShivajiRANS                  = "ShivajiRANS"
	ShivajiComputed              = "ShivajiRANS_computed"
	OneraM6                      = "oneram6"
)

var budgetFieldMap = map[string]string{
	"Source":       "Computed_Source",
	"NuGradMagBar": "NuHatGradNormBar",
	"WallDistance": "WallDist",
}

var lavalLoc string

// All of these assume that the working directory is $GOPATH, which should be set
// from the main script

func GetDatasets(data string, caller driver.Syscaller) ([]ransuq.Dataset, error) {
	var datasets []ransuq.Dataset

	flatplate3_06 := newFlatplate(3e6, 0, "med", "atwall")
	flatplate4_06 := newFlatplate(4e6, 0, "med", "atwall")
	flatplate5_06 := newFlatplate(5e6, 0, "med", "atwall")
	flatplate6_06 := newFlatplate(6e6, 0, "med", "atwall")
	flatplate7_06 := newFlatplate(7e6, 0, "med", "atwall")

	flatplate3_06_BL := newFlatplate(3e6, 0, "med", "justbl")
	flatplate4_06_BL := newFlatplate(4e6, 0, "med", "justbl")
	flatplate5_06_BL := newFlatplate(5e6, 0, "med", "justbl")
	flatplate6_06_BL := newFlatplate(6e6, 0, "med", "justbl")
	flatplate7_06_BL := newFlatplate(7e6, 0, "med", "justbl")

	blIgnoreNames, blIgnoreFunc := GetIgnoreData("justbl")

	// TODO: Move these to a function
	flatplateLoc := filepath.Join(gopath, "data", "ransuq", "flatplate", "med")

	flatplate3_06_budget_BL_Loc := filepath.Join(flatplateLoc, "Flatplate_Re_3e_06", "turb_flatplate_sol_budget.dat")
	flatplate3_06_budget_BL := &datawrapper.CSV{
		Location:    flatplate3_06_budget_BL_Loc,
		Name:        "Flat306Budget",
		IgnoreFunc:  blIgnoreFunc,
		IgnoreNames: blIgnoreNames,
		FieldMap:    budgetFieldMap,
	}

	flatplate5_06_budget_BL_Loc := filepath.Join(flatplateLoc, "Flatplate_Re_5e_06", "turb_flatplate_sol_budget.dat")
	flatplate5_06_budget_BL := &datawrapper.CSV{
		Location:    flatplate5_06_budget_BL_Loc,
		Name:        "Flat506Budget",
		IgnoreFunc:  blIgnoreFunc,
		IgnoreNames: blIgnoreNames,
		FieldMap:    budgetFieldMap,
	}

	flatplate7_06_budget_BL_Loc := filepath.Join(flatplateLoc, "Flatplate_Re_7e_06", "turb_flatplate_sol_budget.dat")
	flatplate7_06_budget_BL := &datawrapper.CSV{
		Location:    flatplate7_06_budget_BL_Loc,
		Name:        "Flat706Budget",
		IgnoreFunc:  blIgnoreFunc,
		IgnoreNames: blIgnoreNames,
		FieldMap:    budgetFieldMap,
	}

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
	case SingleFlatplateBL:
		datasets = []ransuq.Dataset{flatplate5_06_BL}
	case MultiFlatplate:
		datasets = multiFlatplate
	case MultiFlatplateBL:
		datasets = []ransuq.Dataset{flatplate3_06_BL, flatplate5_06_BL, flatplate7_06_BL}
	case ExtraFlatplate:
		datasets = []ransuq.Dataset{newFlatplate(1e6, 0, "med", "atwall"), newFlatplate(2e6, 0, "med", "atwall"), newFlatplate(1.5e6, 0, "med", "atwall")}
	case FlatplateSweep:
		datasets = flatplateSweep
	case FlatplateSweepBl:
		datasets = []ransuq.Dataset{
			flatplate3_06_BL,
			flatplate4_06_BL,
			flatplate5_06_BL,
			flatplate6_06_BL,
			flatplate7_06_BL,
		}
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
		// Need to check correctness of this case
		/*
			case SingleFlatplateBudget:

				location := filepath.Join(gopath, "data", "ransuq", "flatplate", "med", "Flatplate_Re_5e_06", "turb_flatplate_sol_budget.dat")
				datasets = []ransuq.Dataset{
					&datawrapper.CSV{
						Location:    location,
						Name:        "Flat06Budget",
						IgnoreFunc:  func(d []float64) bool { return d[0] < wallDistIgnore },
						IgnoreNames: []string{"WallDist"},
						FieldMap:    budgetFieldMap,
					},
				}
		*/
	case MultiFlatplateBudgetBL:
		datasets = []ransuq.Dataset{
			flatplate3_06_budget_BL,
			flatplate5_06_budget_BL,
			flatplate7_06_budget_BL,
		}
	case DNS5n:
		datasets = []ransuq.Dataset{
			&datawrapper.CSV{
				Location:   filepath.Join(gopath, "data", "ransuq", "HiFi", "exp5xn.txt"),
				Name:       "DNS5n",
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
	case FwNACA0012:
		datasets = []ransuq.Dataset{
			&datawrapper.CSV{
				Location: filepath.Join(gopath, "data", "ransuq", "RANS_Shivaji", "naca0012_fw.dat"),
				Name:     "NACA_0012_Shivaji",
				IgnoreFunc: func([]float64) bool {
					return false
				},
			},
		}
	case FlatPress:
		datasets = []ransuq.Dataset{
			flatplate3_06,
			flatplate4_06,
			flatplate5_06,
			flatplate6_06,
			flatplate6_06,
			newFlatplate(5e6, .30, "med", "atwall"),
			newFlatplate(5e6, .10, "med", "atwall"),
			newFlatplate(5e6, .03, "med", "atwall"),
			newFlatplate(5e6, .01, "med", "atwall"),
			newFlatplate(5e6, 0, "med", "atwall"),
			newFlatplate(5e6, -.01, "med", "atwall"),
			newFlatplate(5e6, -.03, "med", "atwall"),
			newFlatplate(5e6, -.10, "med", "atwall"),
			newFlatplate(5e6, -.30, "med", "atwall"),
		}
	case NacaPressureFlat:
		datasets = []ransuq.Dataset{
			flatplate3_06,
			flatplate4_06,
			flatplate5_06,
			flatplate6_06,
			flatplate7_06,
			newFlatplate(5e6, .30, "med", "atwall"),
			newFlatplate(5e6, .10, "med", "atwall"),
			newFlatplate(5e6, .03, "med", "atwall"),
			newFlatplate(5e6, .01, "med", "atwall"),
			newFlatplate(5e6, -.01, "med", "atwall"),
			newFlatplate(5e6, -.03, "med", "atwall"),
			newFlatplate(5e6, -.10, "med", "atwall"),
			newFlatplate(5e6, -.30, "med", "atwall"),
			newNaca0012(0, "atwall"),
			newNaca0012(1, "atwall"),
			newNaca0012(2, "atwall"),
			newNaca0012(3, "atwall"),
			newNaca0012(4, "atwall"),
			newNaca0012(5, "atwall"),
			newNaca0012(6, "atwall"),
			newNaca0012(7, "atwall"),
			newNaca0012(8, "atwall"),
			newNaca0012(9, "atwall"),
			newNaca0012(10, "atwall"),
			newNaca0012(11, "atwall"),
			newNaca0012(12, "atwall"),
		}
	case NacaPressureFlatBl:
		datasets = []ransuq.Dataset{
			newNaca0012(0, "justbl"),
			newNaca0012(1, "justbl"),
			newNaca0012(2, "justbl"),
			newNaca0012(3, "justbl"),
			newNaca0012(4, "justbl"),
			newNaca0012(5, "justbl"),
			newNaca0012(6, "justbl"),
			newNaca0012(7, "justbl"),
			newNaca0012(8, "justbl"),
			newNaca0012(9, "justbl"),
			newNaca0012(10, "justbl"),
			newNaca0012(11, "justbl"),
			newNaca0012(12, "justbl"),
			flatplate3_06_BL,
			flatplate4_06_BL,
			flatplate5_06_BL,
			flatplate6_06_BL,
			flatplate7_06_BL,
			newFlatplate(5e6, .30, "med", "justbl"),
			newFlatplate(5e6, .10, "med", "justbl"),
			newFlatplate(5e6, .03, "med", "justbl"),
			newFlatplate(5e6, .01, "med", "justbl"),
			newFlatplate(5e6, -.01, "med", "justbl"),
			newFlatplate(5e6, -.03, "med", "justbl"),
			newFlatplate(5e6, -.10, "med", "justbl"),
			newFlatplate(5e6, -.30, "med", "justbl"),
		}
	case PressureBl:
		datasets = []ransuq.Dataset{
			newFlatplate(5e6, .30, "med", "justbl"),
			newFlatplate(5e6, .10, "med", "justbl"),
			newFlatplate(5e6, .03, "med", "justbl"),
			newFlatplate(5e6, .01, "med", "justbl"),
			newFlatplate(5e6, -.01, "med", "justbl"),
			newFlatplate(5e6, -.03, "med", "justbl"),
			newFlatplate(5e6, -.10, "med", "justbl"),
			newFlatplate(5e6, -.30, "med", "justbl"),
		}
	case FlatPressureBl:
		datasets = []ransuq.Dataset{
			flatplate3_06_BL,
			flatplate4_06_BL,
			flatplate5_06_BL,
			flatplate6_06_BL,
			flatplate7_06_BL,
			newFlatplate(5e6, .30, "med", "justbl"),
			newFlatplate(5e6, .10, "med", "justbl"),
			newFlatplate(5e6, .03, "med", "justbl"),
			newFlatplate(5e6, .01, "med", "justbl"),
			newFlatplate(5e6, -.01, "med", "justbl"),
			newFlatplate(5e6, -.03, "med", "justbl"),
			newFlatplate(5e6, -.10, "med", "justbl"),
			newFlatplate(5e6, -.30, "med", "justbl"),
		}
	case SingleNaca0012:
		datasets = []ransuq.Dataset{
			newNaca0012(0, "atwall"),
		}
	case SingleNaca0012Bl:
		datasets = []ransuq.Dataset{
			newNaca0012(0, "justbl"),
		}
	case MultiNaca0012:
		datasets = []ransuq.Dataset{
			newNaca0012(0, "atwall"),
			newNaca0012(3, "atwall"),
			newNaca0012(6, "atwall"),
			newNaca0012(9, "atwall"),
			newNaca0012(12, "atwall"),
		}
	case MultiNaca0012Bl:
		datasets = []ransuq.Dataset{
			newNaca0012(0, "justbl"),
			newNaca0012(3, "justbl"),
			newNaca0012(6, "justbl"),
			newNaca0012(9, "justbl"),
			newNaca0012(12, "justbl"),
		}
	case Naca0012Sweep:
		datasets = []ransuq.Dataset{
			newNaca0012(0, "atwall"),
			newNaca0012(1, "atwall"),
			newNaca0012(2, "atwall"),
			newNaca0012(3, "atwall"),
			newNaca0012(4, "atwall"),
			newNaca0012(5, "atwall"),
			newNaca0012(6, "atwall"),
			newNaca0012(7, "atwall"),
			newNaca0012(8, "atwall"),
			newNaca0012(9, "atwall"),
			newNaca0012(10, "atwall"),
			newNaca0012(11, "atwall"),
			newNaca0012(12, "atwall"),
		}
	case PressureGradientMultiSmall:
		datasets = []ransuq.Dataset{
			newFlatplate(5e6, .30, "med", "atwall"),
			newFlatplate(5e6, .10, "med", "atwall"),
			newFlatplate(5e6, .03, "med", "atwall"),
			newFlatplate(5e6, .01, "med", "atwall"),
			newFlatplate(5e6, 0, "med", "atwall"),
			newFlatplate(5e6, -.01, "med", "atwall"),
			newFlatplate(5e6, -.03, "med", "atwall"),
			newFlatplate(5e6, -.10, "med", "atwall"),
			newFlatplate(5e6, -.30, "med", "atwall"),
		}
	case PressureGradientMulti:
		datasets = []ransuq.Dataset{
			newFlatplate(5e6, 30, "med", "atwall"),
			newFlatplate(5e6, 10, "med", "atwall"),
			newFlatplate(5e6, 3, "med", "atwall"),
			newFlatplate(5e6, 1, "med", "atwall"),
			newFlatplate(5e6, .1, "med", "atwall"),
			newFlatplate(5e6, 0, "med", "atwall"),
			newFlatplate(5e6, -.1, "med", "atwall"),
			newFlatplate(5e6, -1, "med", "atwall"),
			newFlatplate(5e6, -3, "med", "atwall"),
			newFlatplate(5e6, -10, "med", "atwall"),
		}
	case OneraM6:
		datasets = []ransuq.Dataset{
			newOneraM6(),
		}
	case LavalDNS, LavalDNSBL, LavalDNSBLAll, LavalDNSCrop:
		ignoreNames, ignoreFunc := GetIgnoreData(data)
		datasets = []ransuq.Dataset{
			&datawrapper.CSV{
				Location:    lavalLoc,
				Name:        "Laval",
				IgnoreFunc:  ignoreFunc,
				IgnoreNames: ignoreNames,
				FieldMap:    datawrapper.LavalMap,
			},
		}
	case ShivajiRANS:
		ignoreNames, ingoreFunc := GetIgnoreData("atwall")
		datasets = []ransuq.Dataset{
			&datawrapper.CSV{
				Location:    filepath.Join(gopath, "data", "ransuq", "RANS_Shivaji", "bigrans", "data_extracomputed.txt"),
				Name:        "RANS_Shivaji",
				IgnoreFunc:  ingoreFunc,
				IgnoreNames: ignoreNames,
			},
		}
	case ShivajiComputed:
		ignoreNames, ingoreFunc := GetIgnoreData("atwall")
		datasets = []ransuq.Dataset{
			&datawrapper.CSV{
				Location:    filepath.Join(gopath, "data", "ransuq", "RANS_Shivaji", "bigrans", "data_recomputed.txt"),
				Name:        "RANS_Shivaji_Computed",
				IgnoreFunc:  ingoreFunc,
				IgnoreNames: ignoreNames,
			},
		}
	}

	for _, dataset := range datasets {
		if dataset == nil {
			panic("nil dataset")
		}
		fmt.Println(dataset.ID())
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

// re = reynolds number
// cp = coefficient of pressure -- delta p * length * dynamic pressure ()
// uses farfield pressure for zero cp, and symmetry up top for non-zero cp
func newFlatplate(re float64, cp float64, fidelity string, ignoreType string) ransuq.Dataset {
	var basepath, baseconfig string
	flatplateBase := filepath.Join(gopath, "data", "ransuq", "flatplate")
	if cp == 0 {
		basepath = flatplateBase
		baseconfig = filepath.Join(basepath, "base_flatplate_config.cfg")
	} else {
		basepath = filepath.Join(flatplateBase, "pressuregradient")
		baseconfig = filepath.Join(basepath, "base", "base_flatplate_config.cfg")
	}

	restring := strconv.FormatFloat(re, 'g', -1, 64)
	cpstring := strconv.FormatFloat(cp, 'g', -1, 64)

	name := "Flatplate_Re_" + restring
	if cp != 0 {
		name += "_Cp_" + cpstring
	}

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
		panic(err)
	}

	// Load in the existing
	err = drive.LoadFrom(baseconfigFile)
	if err != nil {
		panic(err)
	}

	// set mesh file to be the base mesh file but use relative path

	fullMeshFilename := filepath.Join(flatplateBase, drive.Options.MeshFilename)

	relMeshFilename, err := filepath.Rel(wd, fullMeshFilename)
	if err != nil {
		panic(err)
	}
	drive.Options.MeshFilename = relMeshFilename

	// set other things
	drive.Options.ReynoldsNumber = re

	drive.Options.RefTemperature = drive.Options.FreestreamTemperature

	//get the freestream pressure and density
	gamma := drive.Options.GammaValue
	gasConst := drive.Options.GasConstant
	pressure, density := nondimensionalize.Values(drive.Options.FreestreamTemperature, drive.Options.ReynoldsNumber, drive.Options.MachNumber, gasConst, drive.Options.ReynoldsLength, gamma)
	drive.Options.RefPressure = pressure
	drive.Options.RefDensity = density
	totalT := nondimensionalize.TotalTemperature(drive.Options.FreestreamTemperature, drive.Options.MachNumber, gamma)
	totalP := nondimensionalize.TotalPressure(pressure, drive.Options.MachNumber, gamma)
	totalTString := strconv.FormatFloat(totalT, 'g', 16, 64)
	totalPString := strconv.FormatFloat(totalP, 'g', 16, 64)
	//pString := strconv.FormatFloat(pressure, 'g', 16, 64)
	//drive.Options.MarkerInlet = &su2types.Inlet{"( inlet, " + totalTString + ", " + totalPString + ", 1.0, 0.0, 0.0 )"}
	drive.Options.MarkerInlet = &su2types.Inlet{Strings: []string{"inlet", totalTString, totalPString, "1", "0", "0"}}
	if cp == 0 {
		drive.Options.MarkerOutlet = &su2types.StringDoubleList{
			Strings: []string{"outlet", "farfield"},
			Doubles: []float64{pressure, pressure},
		}
	} else {
		inletVelocity := drive.Options.MachNumber * nondimensionalize.SpeedOfSound(gamma, gasConst, drive.Options.FreestreamTemperature)
		plateLength := 2.0
		dynamicPressure := (1.0 / 2) * density * inletVelocity * inletVelocity * plateLength
		inletPressure := pressure
		outletPressure := inletPressure + cp*dynamicPressure
		drive.Options.MarkerSym = []string{"symmetry", "farfield"}
		drive.Options.MarkerOutlet = &su2types.StringDoubleList{
			Strings: []string{"outlet"},
			Doubles: []float64{outletPressure},
		}
	}
	//drive.Options.MarkerOutlet = &su2types.StringDoubleList{"( outlet, " + pString + ", farfield, " + pString + " )"}

	switch fidelity {
	case "low":
		drive.Options.ResidualReduction = 0.2
	case "med":
		drive.Options.ResidualReduction = 5
		if cp != 0 {
			drive.Options.ResidualReduction = 4
		}
	case "high":
		drive.Options.ResidualReduction = 7
	default:
		panic("bad fidelity")
	}

	//	var ignoreFunc func(d []float64) bool
	//	var ignoreNames []string

	ignoreNames, ignoreFunc := GetIgnoreData(ignoreType)

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

func newOneraM6() ransuq.Dataset {
	basepath := filepath.Join(gopath, "data", "ransuq", "airfoil", "oneram6")
	configName := "turb_ONERAM6.cfg"
	mshName := "mesh_ONERAM6_turb_hexa_43008.su2"
	baseconfig := filepath.Join(basepath, "testcase", configName)
	meshFile := filepath.Join(basepath, "testcase", mshName)

	name := "OneraM6"
	wd := filepath.Join(basepath, "OneraM6")
	drive := &driver.Driver{
		Name:      name,
		Config:    configName,
		Wd:        wd,
		FancyName: "Onera M6",
		Stdout:    name + "_log.txt",
	}

	baseconfigFile, err := os.Open(baseconfig)
	if err != nil {
		panic(err)
	}

	// Set the base config options to be those
	err = drive.LoadFrom(baseconfigFile)
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
		IgnoreFunc:  func(d []float64) bool { return d[0] < wallDistIgnore },
		Name:        name,
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
	err = drive.LoadFrom(baseconfigFile)
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
		IgnoreFunc:  func(d []float64) bool { return d[0] < wallDistIgnore },
		Name:        name,
	}
}

func newNaca0012(aoa float64, ignoreType string) ransuq.Dataset {
	conv := 4.2
	basepath := filepath.Join(gopath, "data", "ransuq", "airfoil", "naca0012")
	configName := "turb_NACA0012.cfg"
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

	fmt.Println(drive.Wd, drive.Config)

	baseconfigFile, err := os.Open(baseconfig)
	if err != nil {
		panic(err)
	}

	// Set the base config options to be those
	err = drive.LoadFrom(baseconfigFile)
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
	drive.Options.ExtIter = 9999
	drive.Options.ResidualReduction = conv

	ignoreName, ignoreFunc := GetIgnoreData(ignoreType)

	// Create an SU2 datawrapper from it
	return &datawrapper.SU2{
		Driver:      drive,
		Su2Caller:   driver.Serial{}, // TODO: Need to figure out how to do this better
		IgnoreNames: ignoreName,
		IgnoreFunc:  ignoreFunc,
		Name:        name,
		ComparisonPostprocessor: datawrapper.AirfoilPostprocessor{},
	}
}

const wallDistIgnore = 1e-10

func GetIgnoreData(ignoreType string) (ignoreNames []string, ignoreFunc func([]float64) bool) {

	lavalIgnoreDist := 3
	switch ignoreType {
	case "laval_dns":
		nX := 2304
		nY := 385
		ignoreNames = []string{"WallDistance", "idx_x", "idx_y", "Source", "NondimSourceUNorm"}
		ignoreFunc = func(d []float64) bool {
			if d[0] < wallDistIgnore {
				return true
			}
			if int(d[1]) > nX-lavalIgnoreDist || int(d[1]) <= lavalIgnoreDist {
				return true
			}
			if int(d[2]) > nY-lavalIgnoreDist || int(d[2]) <= lavalIgnoreDist {
				return true
			}

			return false
		}
	case "laval_dns_crop":
		nX := 2304
		nY := 385
		ignoreNames = []string{"WallDistance", "idx_x", "idx_y", "Chi", "Source", "NuHatGradMagUNorm", "SourceNondimerUNorm", "NondimSourceUNorm"}
		ignoreFunc = func(d []float64) bool {
			if d[0] < wallDistIgnore {
				return true
			}
			idxX := int(d[1])
			idxY := int(d[2])
			chi := d[3]
			source := d[4]
			nugradbar := d[5]
			sourceNondimer := d[6]
			nondimSource := d[7]
			if int(idxX) > nX-lavalIgnoreDist || int(idxX) <= lavalIgnoreDist {
				return true
			}
			if int(idxY) > nY-lavalIgnoreDist || int(idxY) <= lavalIgnoreDist {
				return true
			}
			if chi > 60 || chi < -25 {
				return true
			}
			_ = source
			if nugradbar > 0.9 {
				return true
			}
			if sourceNondimer < 1e-8 {
				return true
			}
			if nondimSource > 20 || nondimSource < -20 {
				return true
			}
			/*
				if source > 0.02 || source < -0.02 {
					return true
				}
			*/

			// Get rid of the two crazy points and their neighbors
			// bad points are {1008, 16} and {1008, 18}
			around := 2
			badPoints := [][2]int{{1008, 16}, {1008, 18}}
			for _, point := range badPoints {
				if idxX >= point[0]-around && idxX <= point[0]+around &&
					idxY >= point[1]-around && idxY <= point[1]+around {
					//fmt.Println("laval load ignore", idxX, idxY, d[3])
					return true
				}
			}

			return false
		}
	case "laval_dns_bl":
		nX := 2304
		nY := 385
		ignoreNames = []string{"WallDistance", "idx_x", "idx_y", "XLoc"}
		ignoreFunc = func(d []float64) bool {
			if d[0] < wallDistIgnore {
				return true
			}
			if d[0] > 1e-2 {
				return true
			}
			if int(d[1]) > nX-lavalIgnoreDist || int(d[1]) <= lavalIgnoreDist {
				return true
			}
			if int(d[2]) > nY-lavalIgnoreDist || int(d[2]) <= lavalIgnoreDist {
				return true
			}
			if d[3] > 3 {
				return true
			}
			return false
		}
	case "laval_dns_bl_all":
		nX := 2304
		nY := 385
		ignoreNames = []string{"WallDistance", "idx_x", "idx_y"}
		ignoreFunc = func(d []float64) bool {
			if d[0] < wallDistIgnore {
				return true
			}
			if d[0] > 1e-2 {
				return true
			}
			if int(d[1]) > nX-lavalIgnoreDist || int(d[1]) <= lavalIgnoreDist {
				return true
			}
			if int(d[2]) > nY-lavalIgnoreDist || int(d[2]) <= lavalIgnoreDist {
				return true
			}
			return false
		}
	case "atwall":
		ignoreNames = []string{"WallDistance"}
		ignoreFunc = func(d []float64) bool { return d[0] < wallDistIgnore }
	case "justbl":
		ignoreNames = []string{"IsInBL", "WallDistance"}
		ignoreFunc = func(d []float64) bool {
			// Ignore if too close to the wall (probably at the wall)
			if d[1] < wallDistIgnore {
				return true
			}

			if !(d[0] == 0 || d[0] == 1) {
				panic("non one or zero boolean")
			}
			return d[0] == 0 // ignore if not in the BL
		}
	default:
		panic("unknown ignore type")
	}
	return ignoreNames, ignoreFunc
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
