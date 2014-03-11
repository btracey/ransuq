package ransuq

import (
	"sync"

	"github.com/reggo/reggo/common"
)

type Dataset interface {
	Load([]string) (common.RowMatrix, error)
	ID() string
}

type Generatable interface {
	Generated() bool
	Run() error
}

// A CompGenerator is a type which can take in the trained algorithm and
// return a Generatable dataset
type Comparable interface {
	Comparison(mlAlgFilename string, outLoc string, featureKind string) (Generatable, error)
}

// A Postprocesser implements special post-processing routines
type PostProcessor interface {
	PostProcess() error
}

//type TestableDataset interface {
//	ML(finishedAlgorithm) GeneratableDataset // Given the answer return something that can be run
//}

func GetGeneratables(datasets []Dataset) []Generatable {
	var gens []Generatable
	for i := range datasets {
		// If it's not a generatable, we can't run it, so ignore this dataset
		generatable, ok := datasets[i].(Generatable)
		if !ok {
			continue
		}
		gens = append(gens, generatable)
	}
	return gens
}

// GenerateData loops over the datasets and runs the datasets which are generatable
// and need to be run. If redo is true, the data will be rerun even if the
// Generatable says it has been generated. If concurrent is true, the data generation
// runs will happen concurrently, otherwise they will happen sequentially
func GenerateData(datasets []Generatable) []error {
	errs := make([]error, len(datasets))
	wg := &sync.WaitGroup{}
	for i, generatable := range datasets {

		// If it's already been run we can skip it unless redo is set
		if generatable.Generated() {
			continue
		}

		f := func(g Generatable, i int) {

			err := g.Run()
			errs[i] = err

			// See if it's a post-processor, and if so, run
			//	p, ok := generatable.(Postprocessor)
			//	if ok {
			//		errs[i] = p.PostProcess()
			//	}
			wg.Done()
		}

		wg.Add(1)
		go f(generatable, i)
	}
	wg.Wait()

	// TODO: Need to do better error-handling stuff
	// If there are any errors, return the error list, otherwise, return nil
	for i := range errs {
		if errs[i] != nil {
			return errs
		}
	}
	return nil
}
