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
	QuickTraining    = "quick"
	FiveKIter        = "5kiter"
	TenKIter         = "10kiter"
	OneHundIter      = "100iter"
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
	case QuickTraining:
		return &ransuq.Trainer{
			TrainSettings: ransuq.TrainSettings{
				ObjAbsTol:   1e-3,
				GradAbsTol:  1e-3,
				MaxFunEvals: 1e2,
			},
			InputScaler:  &scale.Normal{},
			OutputScaler: &scale.Normal{},
			Losser:       loss.SquaredDistance{},
			Regularizer:  nil,
		}, nil
	case StandardTraining:
		return &ransuq.Trainer{
			TrainSettings: ransuq.TrainSettings{
				ObjAbsTol:   1e-6,
				GradAbsTol:  1e-6,
				MaxFunEvals: 3e4,
			},
			InputScaler:  &scale.Normal{},
			OutputScaler: &scale.Normal{},
			Losser:       loss.SquaredDistance{},
			Regularizer:  nil,
		}, nil
	case FiveKIter:
		return &ransuq.Trainer{
			TrainSettings: ransuq.TrainSettings{
				ObjAbsTol:   1e-6,
				GradAbsTol:  1e-6,
				MaxFunEvals: 5e3,
			},
			InputScaler:  &scale.Normal{},
			OutputScaler: &scale.Normal{},
			Losser:       loss.SquaredDistance{},
			Regularizer:  nil,
		}, nil
	case TenKIter:
		return &ransuq.Trainer{
			TrainSettings: ransuq.TrainSettings{
				ObjAbsTol:   1e-6,
				GradAbsTol:  1e-6,
				MaxFunEvals: 1e4,
			},
			InputScaler:  &scale.Normal{},
			OutputScaler: &scale.Normal{},
			Losser:       loss.SquaredDistance{},
			Regularizer:  nil,
		}, nil
	case OneHundIter:
		return &ransuq.Trainer{
			TrainSettings: ransuq.TrainSettings{
				ObjAbsTol:   1e-6,
				GradAbsTol:  1e-6,
				MaxFunEvals: 1e2,
			},
			InputScaler:  &scale.Normal{},
			OutputScaler: &scale.Normal{},
			Losser:       loss.SquaredDistance{},
			Regularizer:  nil,
		}, nil
	}
}
