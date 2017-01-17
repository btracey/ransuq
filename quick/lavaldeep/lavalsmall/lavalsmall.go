package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/btracey/numcsv"
	"github.com/gonum/matrix/mat64"
)

func main() {

	rand.Seed(time.Now().UnixNano())

	nSubSamples := 100000

	newfilename := "laval_csv_" + strconv.Itoa(nSubSamples) + "_2" + ".dat"
	f2, err := os.Create(newfilename)
	if err != nil {
		log.Fatal(err)
	}
	defer f2.Close()

	filename := "/Users/brendan/Documents/mygo/data/ransuq/laval/laval_csv_mldata.dat"
	f, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	r := numcsv.NewReader(f)
	headings, err := r.ReadHeading()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(headings)
	fmt.Println("num headings = ", len(headings))
	allData, err := r.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	rows, cols := allData.Dims()
	fmt.Println("num samples = ", rows, cols)

	perm := rand.Perm(rows)

	newData := mat64.NewDense(nSubSamples, cols, nil)
	row := make([]float64, cols)
	for i := 0; i < nSubSamples; i++ {
		idx := perm[i]
		allData.Row(row, idx)
		newData.SetRow(i, row)
	}

	w := numcsv.NewWriter(f2)
	w.WriteAll(headings, newData)
}
