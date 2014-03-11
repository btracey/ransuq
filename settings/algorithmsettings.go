// Algorithm settings is a list of machine learning algorithm settings to aid
// in creating reproducible results

package settings

import (
	"fmt"
	"sort"

	"github.com/reggo/reggo/nnet"
	regtrain "github.com/reggo/reggo/train"
)

// TODO: Need to solve combinatorics with the different training options (losser, regularizer, etc.)

func init() {
	sortedAlgorithm = append(sortedAlgorithm, NetOneFifty)
	sortedAlgorithm = append(sortedAlgorithm, NetTwoFifty)

	sort.Strings(sortedAlgorithm)
}

const (
	NetOneFifty = "net_1_50"
	NetTwoFifty = "net_2_50"
)

var sortedAlgorithm []string

type Missing struct {
	Prefix  string
	Options []string
}

func (m Missing) Error() string {
	return fmt.Sprintf("%s: acceptable options: %v", m.Prefix, m.Options)
}

// GetTrainer takes in a string and returns a trainable. This is a safe way
// of getting one of the normally-used settings
func getAlgorithm(alg string, inputDim, outputDim int) (regtrain.Trainable, error) {
	switch alg {
	case NetOneFifty:
		return nnet.NewSimpleTrainer(inputDim, outputDim, 1, 50, nnet.Linear{})
	case NetTwoFifty:
		return nnet.NewSimpleTrainer(inputDim, outputDim, 2, 50, nnet.Linear{})
	default:
		return nil, Missing{
			Prefix:  "algorithm setting not found",
			Options: sortedAlgorithm,
		}
	}
}
