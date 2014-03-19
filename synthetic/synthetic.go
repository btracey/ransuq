package synthetic

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"

	"github.com/btracey/ransuq/synthetic/sa"
	"github.com/gonum/matrix/mat64"
	"github.com/reggo/reggo/common"
)

var syntheticDatasetSize int = 1e5

var gopath string

func init() {
	gopath = os.Getenv("GOPATH")
}

var FlatplateBounds = &SABounds{
	Name:        "Flatplate",
	Chi:         [2]float64{50, 400},
	Omegabar:    [2]float64{0, 200000},
	LogWallDist: [2]float64{-6, 0},
	LogNu:       [2]float64{-9, -7},
}

type SABounds struct {
	Name        string
	Chi         [2]float64
	Omegabar    [2]float64
	LogWallDist [2]float64
	LogNu       [2]float64
}

type Production struct {
	Bounds *SABounds
}

func (p Production) ID() string {
	return "SyntheticProduction" + p.Bounds.Name
}

func (p Production) Filename() string {
	return p.ID() + ".csv"
}

// Returns the data path
func (p Production) Path() string {
	return filepath.Join(gopath, "data", "ransuq", "synthetic", "production", p.Bounds.Name)
}

/*
func (p Production) filename() string {
	return filepath.Join(p.Path(), p.ID())
}
*/

func (p Production) Generated() bool {
	// Assume that if the file is there, it has been generated
	_, err := os.Open(filepath.Join(p.Path(), p.Filename()))
	b := !os.IsNotExist(err)
	return b
}

func (p Production) Run() error {
	fmt.Println("In production run")
	chiBounds := p.Bounds.Chi
	omegaBarBounds := p.Bounds.Omegabar
	logWallDistBounds := p.Bounds.LogWallDist
	logNuBounds := p.Bounds.LogNu

	headings := []string{"Chi", "OmegaBar", "SourceNondimer", "Production"}

	// Generate random data
	data := mat64.NewDense(syntheticDatasetSize, 4, nil)
	for i := 0; i < syntheticDatasetSize; i++ {
		chi := rand.Float64()*(chiBounds[1]-chiBounds[0]) + chiBounds[0]
		omegaBar := rand.Float64()*(omegaBarBounds[1]-omegaBarBounds[0]) + omegaBarBounds[0]
		logWallDist := rand.Float64()*(logWallDistBounds[1]-logWallDistBounds[0]) + logWallDistBounds[0]
		logNu := rand.Float64()*(logNuBounds[1]-logNuBounds[0]) + logNuBounds[0]

		sourceNondim := math.Pow(10, logNu-logWallDist) * (1 + chi)
		sourceNondim = sourceNondim * sourceNondim

		data.Set(i, 0, chi)
		data.Set(i, 1, omegaBar)
		data.Set(i, 2, sourceNondim)
		fmt.Println(sourceNondim)
		production := sa.Production(chi, omegaBar, sourceNondim)
		data.Set(i, 3, production)
	}

	return writeCSV(headings, data, p.Path(), p.Filename())
}

func (p Production) Load(fields []string) (common.RowMatrix, error) {
	return readCSV(fields, p.Path(), p.Filename())
}

func writeCSV(headings []string, data *mat64.Dense, path, filename string) error {
	r, c := data.Dims()
	// Now that the data has been generated, save it as a csv file
	records := make([][]string, r)
	for i := range records {
		records[i] = make([]string, c)
		for j := range records[i] {
			records[i][j] = strconv.FormatFloat(data.At(i, j), 'f', 16, 64)
		}
	}

	// Now actually save it as a csv
	err := os.MkdirAll(path, 0700)
	if err != nil {
		return err
	}

	f, err := os.Create(filepath.Join(path, filename))
	if err != nil {
		return err
	}
	defer f.Close()
	enc := csv.NewWriter(f)
	err = enc.Write(headings)
	if err != nil {
		return err
	}
	err = enc.WriteAll(records)
	if err != nil {
		return err
	}
	return nil
}

func readCSV(fields []string, path, filename string) (*mat64.Dense, error) {
	f, err := os.Open(filepath.Join(path, filename))
	if err != nil {
		return nil, err
	}

	enc := csv.NewReader(f)
	records, err := enc.ReadAll()

	// First record is all of the headings
	headings := records[0]

	// make a map of the headings
	m := make(map[string]int)
	for i, head := range headings {
		m[head] = i
	}

	indTrans := make([]int, len(fields))
	for i, field := range fields {
		ind, ok := m[field]
		if !ok {
			return nil, fmt.Errorf("field %v not present", field)
		}
		indTrans[i] = ind
	}

	records = records[1:]

	nRecords := len(records)

	mat := mat64.NewDense(nRecords, len(fields), nil)
	for i := 0; i < nRecords; i++ {
		for j := range indTrans {
			ind := indTrans[j]
			s := records[i][ind]
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return nil, err
			}
			mat.Set(i, j, v)
		}
	}
	return mat, nil
}
