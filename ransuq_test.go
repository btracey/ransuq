package ransuq_test

import (
	"math/rand"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	. "github.com/btracey/ransuq"
	"github.com/btracey/ransuq/settings"
	"github.com/btracey/su2tools/driver"
	"github.com/gonum/blas/cblas"
	"github.com/gonum/matrix/mat64"
	"github.com/reggo/reggo/common"
)

func init() {
	mat64.Register(cblas.Blas{})
	runtime.GOMAXPROCS(runtime.NumCPU())
}

type GeneratableDataset struct {
	Str   string
	Cores int
}

func (t *GeneratableDataset) Load(str []string) (common.RowMatrix, error) {
	_ = Generatable(t)
	nSamples := 10
	m := mat64.NewDense(nSamples, len(str), nil)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < len(str); j++ {
			m.Set(i, j, rand.Float64())
		}
	}
	return m, nil
}

func (t *GeneratableDataset) ID() string {
	return t.Str
}

func (t *GeneratableDataset) Generated() bool {
	return false
}

func (t *GeneratableDataset) NumCores() int {
	return t.Cores
}

func (t *GeneratableDataset) Run() error {
	time.Sleep(time.Duration(rand.Intn(5)) * time.Second)
	return nil
}

func TestMulti(t *testing.T) {
	set, err := settings.GetSettings(
		"multi_flatplate",
		"flatplate_sweep",
		"nondim_production",
		"none",
		"net_2_50",
		settings.StandardTraining,
		driver.Serial{true},
	)
	if err != nil {
		t.Errorf(err.Error())
	}

	var sets []*Settings
	sets = append(sets, set)

	/*
		set, err = settings.GetSettings(
			"multi_flatplate",
			"single_rae",
			"nondim_source",
			"none",
			"net_2_50",
			settings.StandardTraining,
			driver.Serial{true},
		)
		if err != nil {
			t.Errorf(err.Error())
		}
		sets = append(sets, set)
	*/

	setFake := &Settings{
		FeatureSet: "Test1",
		TrainingData: []Dataset{
			&GeneratableDataset{"test1_str1", 1},
			&GeneratableDataset{"test1_str2", 6},
		},
		InputFeatures:  []string{"feat1", "feat2"},
		OutputFeatures: []string{"outfeat_1"},
		WeightFeatures: nil,
		WeightFunc:     nil,
		Savepath:       filepath.Join("Users", "brendan", "Documents", "mygo", "test", "ransuq"),
	}

	setFake.Trainer, err = settings.GetTrainer(settings.StandardTraining, "net_2_50", 2, 1)

	sets = append(sets, setFake)

	/*
			set, err = settings.GetSettings(
				"extra_flatplate",
				"single_rae",
				"nondim_source",
				"none",
				"net_2_50",
				settings.StandardTraining,
				driver.Serial{true},
			)
			if err != nil {
				t.Errorf(err.Error())
			}
			sets = append(sets, set)


		for _, data := range set.TrainingData {
			su := data.(*datawrapper.SU2)
			su.Driver.Options.ResidualReduction = 1
		}
	*/

	errs := MultiTurb(sets, NewLocalScheduler())
	for i, err := range errs {
		if err != nil {
			t.Errorf("learner %v: %v", i, err)
		}
	}
	/*
		err = os.Remove(filepath.Join(setFake.Savepath, "trained_algorithm.json"))
		if err != nil {
			t.Errorf("err deleting: %v", err)
		}
	*/
}
