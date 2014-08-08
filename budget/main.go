package main

import (
	"fmt"
	"log"
	"os"

	"github.com/btracey/ransuq/dataloader"
	"github.com/btracey/su2tools/mesh"
	"github.com/davecheney/profile"
	"github.com/gonum/matrix/mat64"
)

func main() {
	defer profile.Start(profile.CPUProfile).Stop()

	dataset := "/Users/brendan/Documents/mygo/data/ransuq/flatplate/med/Flatplate_Re_3e_06_test/turb_flatplate_sol.dat"
	meshfilename := "/Users/brendan/Documents/mygo/data/ransuq/flatplate/mesh_flatplate_turb_137x97.su2"

	// Load in SU2 Data
	features := []string{"Source", "Nu", "NuHat", "XLoc", "YLoc", "PointID"}

	srcIdx := findStringLocation(features, "Source")
	pointIdx := findStringLocation(features, "PointID")

	set := &dataloader.Dataset{
		Name:     "Data",
		Filename: dataset,
		Format:   &dataloader.SU2_restart_2dturb{},
	}

	allData, err := dataloader.Load(features, []*dataloader.Dataset{set})
	if err != nil {
		log.Fatal(err)
	}
	data := allData[0] // just limit it to this dataset

	source := make([]float64, len(data))
	for i, pt := range data {
		source[i] = pt[srcIdx]
	}

	dataIdxMap := make(map[mesh.PointID]int)
	for i, pt := range data {
		id := pt[pointIdx]
		dataIdxMap[mesh.PointID(id)] = i
	}

	// Load in the mesh file so there is connectivity information
	f, err := os.Open(meshfilename)
	if err != nil {
		log.Fatal(err)
	}

	msh := &mesh.SU2{}
	_, err = msh.ReadFrom(f)
	if err != nil {
		log.Fatal(err)
	}

	// Compute first derivatives at all the points
	quantityIdxs := []int{srcIdx}

	derivatives := make([][][]float64, len(msh.Points))
	for k := 0; k < len(msh.Points); k++ {
		//fmt.Println("k =", k)
		// Try to compute the budget
		id := msh.Points[k].Id // random point for now
		/*
			fmt.Println("My point location is ", msh.Points[id].Location)
			for pointID := range msh.Points[id].NeighborIDs {
				fmt.Println("Neighbor location is ", msh.Points[pointID].Location)
			}
		*/

		nNeighbors := len(msh.Points[id].NeighborIDs)
		neighborLocs := make([][]float64, nNeighbors)
		neigborValues := make([][]float64, nNeighbors)
		for i := 0; i < nNeighbors; i++ {
			neigborValues[i] = make([]float64, len(quantityIdxs))
		}
		count := 0
		for pointID := range msh.Points[id].NeighborIDs {
			neighborLocs[count] = msh.Points[pointID].Location
			idx, ok := dataIdxMap[pointID]
			if !ok {
				panic(fmt.Sprintf("point %d not found", pointID))
			}
			for j := 0; j < len(quantityIdxs); j++ {
				neigborValues[count][j] = data[idx][quantityIdxs[j]]
			}
			count++
		}

		thisIdx, ok := dataIdxMap[id]
		if !ok {
			panic("this point not known")
		}
		thisVal := make([]float64, len(quantityIdxs))
		for j := 0; j < len(quantityIdxs); j++ {
			thisVal[j] = data[thisIdx][quantityIdxs[j]]
		}
		derivatives[k] = estimateGradients(msh.Points[id].Location, thisVal, neighborLocs, neigborValues)
		//fmt.Println(derivatives[k])
	}
}

func findStringLocation(s []string, str string) int {
	for i, v := range s {
		if v == str {
			return i
		}
	}
	return -1
}

// estimate gradient
func estimateGradients(pointLocation []float64, pointValue []float64, altLocations [][]float64, altValues [][]float64) (derivative [][]float64) {
	// compute all the distances
	/*
		dists := make([]float64, len(altLocations))
		tmp := make([]float64, pointLocation)
		for i := range altLocations {
			floats.SubTo(tmp, pointLocation, altLocations[i])
			dists[i] = floats.Norm(tmp, 2)
		}
	*/
	// Want to fix A_-1 * 1 A_0 x_0 + A_1 x_1 + ... = value
	nDim := len(pointLocation)
	nSamples := len(altLocations)
	nOutputs := len(altValues[0])

	A := mat64.NewDense(nSamples+1, nDim+1, nil)
	for i := 0; i < nSamples; i++ {
		A.Set(i, 0, 1)
		for j := 0; j < nDim; j++ {
			A.Set(i, j+1, altLocations[i][j])
		}
	}
	// Set the local point
	A.Set(nSamples, 0, 1)
	for j := 0; j < nDim; j++ {
		A.Set(nSamples, j+1, pointLocation[j])
	}

	b := mat64.NewDense(nSamples+1, nOutputs, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nOutputs; j++ {
			b.Set(i, j, altValues[i][j])
		}
	}
	for j := 0; j < nOutputs; j++ {
		b.Set(nSamples, j, pointValue[j])
	}

	// Solve
	x := mat64.Solve(A, b)

	// Pull out the derivative
	derivative = make([][]float64, nOutputs)
	for i := 0; i < nOutputs; i++ {
		derivative[i] = make([]float64, len(pointLocation))
		for j := 0; j < len(pointLocation); j++ {
			derivative[i][j] = x.At(j+1, i)
		}
	}
	return derivative
}

/*
// knn finds the k points closest to the x, y data specified
func knn(k int, x, y float64, xdata, ydata []float64) []int {
	dists := make([]float64, len(xdata))
	inds := make([]int, len(xdata))
	for i := range xdata {
		dists[i] = (x-xdata[i])*(x-xdata[i]) + (y-ydata[i])*(y-ydata[i])
	}
	floats.Sort
}
*/
