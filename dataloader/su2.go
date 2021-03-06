package dataloader

import (
	"bytes"
	"encoding/csv"
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"github.com/btracey/su2tools/remove_whitespace"
	"github.com/btracey/turbulence/sa"
)

var suWallDistance string = "WallDist"
var suDensity string = "Conservative_1"
var suRhoU string = "Conservative_2"
var suRhoV string = "Conservative_3"
var suKinematicViscosity string = "KinematicViscosity"
var suTurbKinematicViscosity string = "NuTilde"
var suDUDX string = "DU_0DX_0"
var suDUDY string = "DU_0DX_1"
var suDVDX string = "DU_1DX_0"
var suDVDY string = "DU_1DX_1"
var suDNuHatDX string = "DNuTildeDX_0"
var suDNuHatDY string = "DNuTildeDX_1"
var suSource string = "Residual_3"

const nondimeps = 1e-8

var suMap map[string]*FieldTransformer = map[string]*FieldTransformer{
	"XLoc": &FieldTransformer{
		InternalNames: []string{"x"},
		Transformer:   identityFunc,
	},
	"YLoc": &FieldTransformer{
		InternalNames: []string{"y"},
		Transformer:   identityFunc,
	},
	"Production": &FieldTransformer{
		InternalNames: []string{"Residual_0"},
		Transformer:   identityFunc,
	},
	"Destruction": &FieldTransformer{
		InternalNames: []string{"Residual_1"},
		Transformer:   identityFunc,
	},
	"CrossProduction": &FieldTransformer{
		InternalNames: []string{"Residual_2"},
		Transformer:   identityFunc,
	},
	"PointID": &FieldTransformer{
		InternalNames: []string{"PointID"},
		Transformer:   identityFunc,
	},
	"Nu": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity},
		Transformer:   identityFunc,
	},
	"NuHat": &FieldTransformer{
		InternalNames: []string{suTurbKinematicViscosity},
		Transformer:   identityFunc,
	},
	"WallDistance": &FieldTransformer{
		InternalNames: []string{suWallDistance},
		Transformer:   identityFunc,
	},
	"DNuHatDX": &FieldTransformer{
		InternalNames: []string{suDNuHatDX},
		Transformer:   identityFunc,
	},
	"DNuHatDY": &FieldTransformer{
		InternalNames: []string{suDNuHatDY},
		Transformer:   identityFunc,
	},
	"DUDX": &FieldTransformer{
		InternalNames: []string{suDUDX},
		Transformer:   identityFunc,
	},
	"DUDY": &FieldTransformer{
		InternalNames: []string{suDUDY},
		Transformer:   identityFunc,
	},
	"DVDX": &FieldTransformer{
		InternalNames: []string{suDVDX},
		Transformer:   identityFunc,
	},
	"DVDY": &FieldTransformer{
		InternalNames: []string{suDVDY},
		Transformer:   identityFunc,
	},
	"DUDXBar": &FieldTransformer{
		InternalNames: []string{"OmegaNondimer", suDUDX},
		Transformer:   nondimensionalizer,
	},
	"DUDYBar": &FieldTransformer{
		InternalNames: []string{"OmegaNondimer", suDUDY},
		Transformer:   nondimensionalizer,
	},
	"DVDXBar": &FieldTransformer{
		InternalNames: []string{"OmegaNondimer", suDVDX},
		Transformer:   nondimensionalizer,
	},
	"DVDYBar": &FieldTransformer{
		InternalNames: []string{"OmegaNondimer", suDVDY},
		Transformer:   nondimensionalizer,
	},
	"YPlus": &FieldTransformer{
		InternalNames: []string{"Y_Plus"},
		Transformer:   identityFunc,
	},
	"Source": &FieldTransformer{
		InternalNames: []string{"Residual_3"},
		Transformer:   identityFunc,
	},
	"Density": &FieldTransformer{
		InternalNames: []string{suDensity},
		Transformer:   identityFunc,
	},
	"UVel": &FieldTransformer{
		InternalNames: []string{suDensity, suRhoU},
		Transformer: func(d []float64) (float64, error) {
			if len(d) != 2 {
				return math.NaN(), fmt.Errorf("wrong number of inputs")
			}
			return d[1] / d[0], nil
		},
	},
	"VVel": &FieldTransformer{
		InternalNames: []string{suDensity, suRhoV},
		Transformer: func(d []float64) (float64, error) {
			if len(d) != 2 {
				return math.NaN(), fmt.Errorf("wrong number of inputs")
			}
			return d[1] / d[0], nil
		},
	},
	"Viscosity": &FieldTransformer{
		InternalNames: []string{suDensity, suKinematicViscosity},
		Transformer: func(d []float64) (float64, error) {
			if len(d) != 2 {
				return math.NaN(), fmt.Errorf("wrong number of inputs")
			}
			return d[1] * d[0], nil
		},
	},
	"Chi": &FieldTransformer{
		InternalNames: []string{"Chi"},
		Transformer:   identityFunc,
	},
	"Chi_Log": &FieldTransformer{
		InternalNames: []string{"Chi"},
		Transformer:   logFunc,
	},
	"NondimProduction": &FieldTransformer{
		InternalNames: []string{"NondimResidual_0"},
		Transformer:   identityFunc,
	},
	"MulProduction": &FieldTransformer{
		InternalNames: []string{"Mul_Production"},
		Transformer:   identityFunc,
	},
	"NondimProductionMod": &FieldTransformer{
		InternalNames: []string{"NondimResidual_0", suWallDistance, "Residual_0"},
		Transformer: func(d []float64) (float64, error) {
			if len(d) != 3 {
				return math.NaN(), fmt.Errorf("wrong number of inputs")
			}
			nondim := d[0]
			dist := d[1]
			dim := d[2]

			if dist > 1e-4 && dim < 1e-10 {
				return dim, nil
			}
			return nondim, nil
		},
	},
	"NondimDestruction": &FieldTransformer{
		InternalNames: []string{"NondimResidual_1"},
		Transformer:   identityFunc,
	},
	"MulDestruction": &FieldTransformer{
		InternalNames: []string{"Mul_Destruction"},
		Transformer:   identityFunc,
	},
	"NondimDestructionMod": &FieldTransformer{
		InternalNames: []string{"NondimResidual_1", suWallDistance, "Residual_1"},
		Transformer: func(d []float64) (float64, error) {
			if len(d) != 3 {
				return math.NaN(), fmt.Errorf("wrong number of inputs")
			}
			nondim := d[0]
			dist := d[1]
			dim := d[2]

			if dist > 1e-4 && dim < 1e-10 {
				return dim, nil
			}
			return nondim, nil
		},
	},
	"NondimCrossProduction": &FieldTransformer{
		InternalNames: []string{"NondimResidual_2"},
		Transformer:   identityFunc,
	},
	"MulCrossProduction": &FieldTransformer{
		InternalNames: []string{"Mul_CrossProduction"},
		Transformer:   identityFunc,
	},
	"NondimCrossProductionMod": &FieldTransformer{
		InternalNames: []string{"NondimResidual_2", suWallDistance, "Residual_2"},
		Transformer: func(d []float64) (float64, error) {
			if len(d) != 3 {
				return math.NaN(), fmt.Errorf("wrong number of inputs")
			}
			nondim := d[0]
			dist := d[1]
			dim := d[2]

			if dist > 1e-4 && dim < 1e-10 {
				return dim, nil
			}
			return nondim, nil
		},
	},
	"NondimSource": &FieldTransformer{
		InternalNames: []string{"NondimResidual_3"},
		Transformer:   identityFunc,
	},
	"NondimSourceMod": &FieldTransformer{
		InternalNames: []string{"NondimResidual_3", suWallDistance, "Residual_3"},
		Transformer: func(d []float64) (float64, error) {
			if len(d) != 3 {
				return math.NaN(), fmt.Errorf("wrong number of inputs")
			}
			nondim := d[0]
			dist := d[1]
			dim := d[2]

			if dist > 1e-4 && dim < 1e-10 {
				return dim, nil
			}
			return nondim, nil
		},
	},
	"SourceNondimer": &FieldTransformer{
		InternalNames: []string{"SourceNondimer"},
		Transformer:   identityFunc,
	},
	"SourceComputed": &FieldTransformer{
		InternalNames: []string{"Computed_Source"},
		Transformer:   identityFunc,
	},
	"OmegaBar": &FieldTransformer{
		InternalNames: []string{"OmegaBar"},
		Transformer:   identityFunc,
	},
	"OmegaBar_Log": &FieldTransformer{
		InternalNames: []string{"OmegaBar"},
		Transformer:   logFunc,
	},
	"DNuHatDXBar": &FieldTransformer{
		InternalNames: []string{"SourceNondimer", "DNuTildeDX_0"},
		Transformer: func(s []float64) (float64, error) {
			if len(s) != 2 {
				return math.NaN(), errors.New("wrong number of options")
			}
			return s[1] / math.Sqrt(s[0]), nil
		},
	},
	"DNuHatDYBar": &FieldTransformer{
		InternalNames: []string{"SourceNondimer", "DNuTildeDX_1"},
		Transformer: func(s []float64) (float64, error) {
			if len(s) != 2 {
				return math.NaN(), errors.New("wrong number of options")
			}
			return s[1] / math.Sqrt(s[0]), nil
		},
	},
	"NuGradMag": &FieldTransformer{
		InternalNames: []string{"NuHatGradNorm"},
		Transformer:   identityFunc,
	},
	"NuGradMagBar": &FieldTransformer{
		InternalNames: []string{"NuHatGradNormBar"},
		Transformer:   identityFunc,
	},
	"NuGradMagBar_Log": &FieldTransformer{
		InternalNames: []string{"NuHatGradNormBar"},
		Transformer:   logFunc,
	},
	"OmegaNondimer": &FieldTransformer{
		InternalNames: []string{"OmegaNondimer"},
		Transformer:   identityFunc,
	},
	"NuGradNondimer": &FieldTransformer{
		InternalNames: []string{"NuGradNondimer"},
		Transformer:   identityFunc,
	},
	"IsInBL": &FieldTransformer{
		InternalNames: []string{"IsInBL"},
		Transformer:   identityFunc,
	},
	"Fw": &FieldTransformer{
		InternalNames: []string{"Fw"},
		Transformer:   identityFunc,
	},
	"ChiAlt": &FieldTransformer{
		InternalNames: []string{"Chi"},
		Transformer: func(s []float64) (float64, error) {
			return sa.ChiAlt(s[0]), nil
		},
	},
	"NuHatAlt": &FieldTransformer{
		InternalNames: []string{suTurbKinematicViscosity, suKinematicViscosity},
		Transformer: func(s []float64) (float64, error) {
			nu := s[1]
			nuhat := s[0]
			return sa.NuHatAlt(nu, nuhat), nil
		},
	},
	"OmegaAlt": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, "OmegaBar", "OmegaNondimer"},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			omega := s[1] * s[2]
			return sa.OmegaAlt(omega, nu), nil
		},
	},
	"NondimOmegaAlt": &FieldTransformer{
		InternalNames: []string{"OmegaNondimer", "OmegaBar", suTurbKinematicViscosity, suWallDistance, suKinematicViscosity},
		Transformer: func(s []float64) (float64, error) {
			nu := s[4]
			nuhat := s[2]
			omega := s[0] * s[1]
			dist := s[3]
			return sa.NondimOmegaAlt(omega, nu, nuhat, dist), nil
		},
	},
	"NondimOmegaAltStar": &FieldTransformer{
		InternalNames: []string{"OmegaNondimer", "OmegaBar", suTurbKinematicViscosity, suWallDistance, suKinematicViscosity},
		Transformer: func(s []float64) (float64, error) {
			nu := s[4]
			nuhat := s[2]
			omega := s[0] * s[1]
			dist := s[3]
			return sa.NondimOmegaAltStar(omega, nu, nuhat, dist), nil
		},
	},
	"OmegaNondimerAltStar": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, suTurbKinematicViscosity, suWallDistance},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			nuhat := s[1]
			dist := s[2]
			return sa.OmegaNondimerAltStar(nu, nuhat, dist), nil
		},
	},
	"InvOmegaNondimerAltStar": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, suTurbKinematicViscosity, suWallDistance},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			nuhat := s[1]
			dist := s[2]
			return sa.InvOmegaNondimerAltStar(nu, nuhat, dist), nil
		},
	},
	"OmegaAltNondimRatio": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, suTurbKinematicViscosity, suWallDistance},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			nuhat := s[1]
			dist := s[2]

			return sa.OmegaAltNondimRatio(nu, nuhat, dist), nil
		},
	},
	"LogOmegaAltNondimRatio": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, suTurbKinematicViscosity, suWallDistance},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			nuhat := s[1]
			dist := s[2]

			return sa.LogOmegaAltNondimRatio(nu, nuhat, dist), nil
		},
	},
	"SourceAltNondimRatio": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, suTurbKinematicViscosity, suWallDistance},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			nuhat := s[1]
			dist := s[2]
			return sa.SourceAltNondimRatio(nu, nuhat, dist), nil
		},
	},
	"LogSourceAltNondimRatio": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, suTurbKinematicViscosity, suWallDistance},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			nuhat := s[1]
			dist := s[2]
			return sa.LogSourceAltNondimRatio(nu, nuhat, dist), nil
		},
	},
	"SourceNondimerAlt": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, suTurbKinematicViscosity, suWallDistance},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			nuhat := s[1]
			dist := s[2]
			return sa.SourceNondimerAlt(nu, nuhat, dist), nil
		},
	},
	"SourceNondimerAltStar": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, suTurbKinematicViscosity, suWallDistance},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			nuhat := s[1]
			dist := s[2]
			return sa.SourceNondimerAltStar(nu, nuhat, dist), nil
		},
	},
	"InvSourceNondimerAltStar": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, suTurbKinematicViscosity, suWallDistance},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			nuhat := s[1]
			dist := s[2]
			return sa.InvSourceNondimerAltStar(nu, nuhat, dist), nil
		},
	},
	"SourceAlt": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, suSource},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			source := s[1]
			return sa.SourceAlt(source, nu), nil
		},
	},
	"NondimSourceAltStar": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, suTurbKinematicViscosity, suWallDistance, suSource},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			nuhat := s[1]
			dist := s[2]
			source := s[3]
			return sa.NondimSourceAltStar(source, nu, nuhat, dist), nil
		},
	},
	"NuGradMagAlt": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, "NuHatGradNorm"},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			grad := s[1]
			return sa.NuGradMagAlt(grad, nu), nil
		},
	},
	"NondimNuGradMagAlt": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, suTurbKinematicViscosity, suWallDistance, "NuHatGradNorm"},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			nuhat := s[1]
			dist := s[2]
			grad := s[3]
			return sa.NondimNuGradMagAlt(grad, nu, nuhat, dist), nil
		},
	},
	"NondimNuGradMagAltStar": &FieldTransformer{
		InternalNames: []string{suKinematicViscosity, suTurbKinematicViscosity, suWallDistance, "NuHatGradNorm"},
		Transformer: func(s []float64) (float64, error) {
			nu := s[0]
			nuhat := s[1]
			dist := s[2]
			grad := s[3]
			return sa.NondimNuGradMagAltStar(grad, nu, nuhat, dist), nil
		},
	},

	/*
		"Fw": &FieldTransformer{
			InternalNames: fullSANames,
			Transformer: func(s []float64) (float64, error) {
				t := sa.SA{
					NDim:                2,
					Nu:                  s[0],
					NuHat:               s[1],
					DNuHatDX:            []float64{s[2], s[3]},
					DUIdXJ:              [][]float64{{s[4], s[5]}, {s[6], s[7]}},
					WallDistance:        s[8],
					LimitedWallDistance: 1e-10,
				}
				_ = t.Source()
				return t.Fw, nil
			},
		},
	*/
}

var nuair = 1.48e-5

var fullSANames = []string{suKinematicViscosity, suTurbKinematicViscosity, "DNuTildeDX_1", "DNuTildeDX_0", suDUDX, suDUDY, suDVDX, suDVDY, suWallDistance}

// Divide to non-dimensionalize, multiply to redimensionalize
func nuscale(nu, nuhat float64) float64 {
	return nu + nuhat
}

func sourcescale(nu, nuhat, omega float64) float64 {
	return omega*(nu+nuhat) + 1e-6
}

func omegascale(nu, nuhat, omega float64) float64 {
	return sourcescale(nu, nuhat, omega) / nuscale(nu, nuhat)
}

func nuhatthresh(nu, nuhat float64) float64 {
	return (nuhat + 3*nu) / (nuair / nu)
}

func distthresh(dist, nu float64) float64 {
	return math.Min(dist, 100*3*nu)
}

// SU2 is a type for data from an su2_restart restart file
type SU2_restart_2dturb struct {
	ReferenceLength float64
}

// Fieldmap specifies which dataset fieldnames are needed to get that fieldname
func (s *SU2_restart_2dturb) Fieldmap(fieldname string) *FieldTransformer {
	// Here is where to put in length scale
	return suMap[fieldname]
}

func (s *SU2_restart_2dturb) NewAppendFields(filename string, newFilename string, newVarnames []string, newData [][]float64) error {
	fmt.Println("in su2 new append fields")

	// Check inputs
	nNewVars := len(newVarnames)
	for _, point := range newData {
		if len(point) != nNewVars {
			return fmt.Errorf("New data length doesn't match")
		}
	}

	// Open file
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	buf := &bytes.Buffer{}

	// Read all the bytes from the file
	byt, err := ioutil.ReadAll(file)
	if err != nil {
		return err
	}

	//Remove trailing whitespace if it exists
	remove_whitespace.RemoveTrailingWhitespace(byt, buf)

	reader := csv.NewReader(buf)

	// Read in records and append new varnames
	headingRecord, err := s.readHeadings(reader)
	if err != nil {
		return fmt.Errorf("error reading headings: " + err.Error())
	}

	headingRecord = append(headingRecord, newVarnames...)

	// All other rows are tab delimited
	otherRecords, err := reader.ReadAll()
	if err != nil {
		return fmt.Errorf("error reading all: " + err.Error())
	}
	for i := range otherRecords {
		for _, f := range newData[i] {
			//otherRecords[i] = otherRecords[i][:len(otherRecords[i])-1]
			//fmt.Println(f)
			otherRecords[i] = append(otherRecords[i], strconv.FormatFloat(f, 'e', 16, 64))
		}
	}
	// Make the new path if necessary
	dir, _ := filepath.Split(newFilename)
	err = os.MkdirAll(dir, 0700)
	if err != nil {
		return err
	}

	// Now print the new data
	newfile, err := os.Create(newFilename)
	if err != nil {
		return err
	}
	defer newfile.Close()

	// Need to custom write all of the headings
	headingBytes := make([]byte, 0)
	for i := 0; i < len(headingRecord); i++ {
		headingBytes = append(headingBytes, '"')
		headingBytes = append(headingBytes, headingRecord[i]...)
		headingBytes = append(headingBytes, '"')
		if i != len(headingRecord) {
			headingBytes = append(headingBytes, '\t')
		}
	}
	headingBytes = append(headingBytes, '\n')
	newfile.Write(headingBytes)

	writer := csv.NewWriter(newfile)
	writer.Comma = '\t'
	//err = writer.Write(headingRecord)
	//if err != nil {
	//	return err
	//}
	err = writer.WriteAll(otherRecords)
	if err != nil {
		return fmt.Errorf("error writing fields: " + err.Error())
	}
	return nil
}

func (s *SU2_restart_2dturb) readHeadings(reader *csv.Reader) ([]string, error) {
	reader.Comma = '\t'
	// First row is comma delimited
	return reader.Read()
}

func (s *SU2_restart_2dturb) ReadFields(fields []string, filename string) ([][]float64, error) {
	// Read in the file and get all of the fields
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	buf := &bytes.Buffer{}

	// Read all the bytes from the file
	byt, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	//Remove trailing whitespace if it exists
	remove_whitespace.RemoveTrailingWhitespace(byt, buf)

	reader := csv.NewReader(buf)

	record, err := s.readHeadings(reader)
	if err != nil {
		return nil, err
	}

	// Map headerToColumn
	headingToColumn := make(map[string]int)
	for i := range record {
		headingToColumn[record[i]] = i
	}

	// Check that all the fields have headings and make
	for _, field := range fields {
		_, ok := headingToColumn[field]
		if !ok {
			return nil, fmt.Errorf("Field %s does not exist in the file", field)
		}
	}

	// All other rows are tab delimited
	otherRecords, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("error reading all: " + err.Error())
	}

	data := make([][]float64, len(otherRecords))
	for i := range data {
		data[i] = make([]float64, len(fields))
	}

	for j, field := range fields {
		// Get the column of the field in the record
		col, ok := headingToColumn[field]
		if !ok {
			panic("Shouldn't be here")
		}
		for i, record := range otherRecords {
			data[i][j], err = strconv.ParseFloat(record[col], 64)
			if err != nil {
				return nil, err
			}
		}
	}

	return data, nil
}
