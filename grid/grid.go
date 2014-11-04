package grid

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

// Grid selects at most one data point from the grid defined by the locations
// Ar error is returned if it is outside the bounds.
func Grid(data *mat64.Dense, bounds [][]float64) ([]int, error) {
	nSamples, nDim := data.Dims()
	if len(bounds) != nDim {
		panic("grid: length of bounds is not equal to number of dimensions")
	}

	for _, bound := range bounds {
		if !sort.Float64sAreSorted(bound) {
			panic("bound isn't sorted")
		}
	}

	perm := rand.Perm(nSamples)

	m := make(map[string]int)

	box := make([]int, nDim)
	for i := 0; i < nSamples; i++ {
		elem := perm[i]
		for j := 0; j < nDim; j++ {
			bound := bounds[j]
			v := data.At(elem, j)
			idx := -1
			for k := 0; k < len(bound)-1; k++ {
				if v >= bound[k] && v <= bound[k+1] {
					idx = k
					break
				}
			}
			if idx == -1 {
				fmt.Println(v, bound)
				return nil, errors.New("grid: point out of bounds")
			}
			box[j] = idx
		}
		id := BoxToString(box)
		m[id] = elem
	}

	idxs := make([]int, len(m))
	var count int
	// Find the unique indices
	for _, val := range m {
		//fmt.Println(key)
		idxs[count] = val
		count++
	}

	return idxs, nil
}

func BoxToString(box []int) string {
	str := strconv.Itoa(box[0])
	for i := 1; i < len(box); i++ {
		str += "_" + strconv.Itoa(box[i])
	}
	return str
}

type BoundKind int

const (
	// Use the quantiles of the CDF
	QuantileBounds BoundKind = iota
)

// TODO: make nPoints a slice
func FindBounds(kind BoundKind, data *mat64.Dense, nPoints int) [][]float64 {
	rows, cols := data.Dims()
	bounds := make([][]float64, cols)
	for i := range bounds {
		bounds[i] = make([]float64, nPoints+1)
	}
	switch kind {
	case QuantileBounds:
		col := make([]float64, rows)
		quantiles := make([]float64, nPoints+1)
		floats.Span(quantiles, 0, 1)
		quantiles = quantiles[1 : len(quantiles)-1]
		for i := 0; i < cols; i++ {
			data.Col(col, i)
			sort.Float64s(col)
			//fmt.Println("i = ", i)
			//fmt.Println("first 10", col[:10])
			//fmt.Println("last 10", col[len(col)-10:])
			bounds[i][0] = math.Inf(-1)
			for j, q := range quantiles {
				bounds[i][j+1] = stat.Quantile(q, stat.Empirical, col, nil)
			}
			bounds[i][len(bounds[i])-1] = math.Inf(1)
		}
	}
	return bounds
}
