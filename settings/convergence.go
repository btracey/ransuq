package settings

import (
	"sort"

	"github.com/btracey/ransuq"

	"github.com/reggo/reggo/loss"
	"github.com/reggo/reggo/scale"
)

func init() {
	sortedConvergence = append(sortedConvergence, StandardTraining)

	sort.Strings(sortedConvergence)
}

var sortedConvergence []string

const (
	// TODO: Better name
	StandardTraining = "standard"
)

func GetTrainer(train string, algorithm string, inputDim, outputDim int) (*ransuq.Trainer, error) {
	trainer, err := getTrainSettings(train)
	if err != nil {
		return nil, err
	}
	alg, err := getAlgorithm(algorithm, inputDim, outputDim)
	if err != nil {
		return nil, err
	}
	trainer.Algorithm = alg
	return trainer, nil
}

// Returns a trainer extecpt for the algorithm
func getTrainSettings(train string) (*ransuq.Trainer, error) {
	switch train {
	default:
		return nil, Missing{
			Prefix:  "convergence setting not found",
			Options: sortedConvergence,
		}
	case StandardTraining:
		return &ransuq.Trainer{
			TrainSettings: ransuq.TrainSettings{
				ObjAbsTol:   1e-6,
				GradAbsTol:  1e-6,
				MaxFunEvals: 1e5,
			},
			InputScaler:  &scale.Normal{},
			OutputScaler: &scale.Normal{},
			Losser:       loss.SquaredDistance{},
			Regularizer:  nil,
		}, nil
	}
}
