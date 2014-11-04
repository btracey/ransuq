// Algorithm settings is a list of machine learning algorithm settings to aid
// in creating reproducible results

package settings

import (
	"fmt"
	"sort"

	"github.com/btracey/ransuq/mlalg"
	"github.com/reggo/reggo/supervised/nnet"
	regtrain "github.com/reggo/reggo/train"
)

// TODO: Need to solve combinatorics with the different training options (losser, regularizer, etc.)

func init() {
	sortedAlgorithm = append(sortedAlgorithm, NetOneFifty)
	sortedAlgorithm = append(sortedAlgorithm, NetTwoFifty)

	sort.Strings(sortedAlgorithm)
}

const (
	NetOneFifty         = "net_1_50"
	NetTwoFifty         = "net_2_50"
	NetThreeFifty       = "net_3_50"
	NetTwoHundred       = "net_2_100"
	MulNetTwoTwentyFive = "mul_net_2_25"
	MulNetTwoThirtyFive = "mul_net_2_35"
	MulNetTwoFourty     = "mul_net_2_40"
	MulNetTwoFourtyFive = "mul_net_2_45"
	MulNetTwoFifty      = "mul_net_2_50"
	MulNetThreeFifty    = "mul_net_3_50"
	MulNetTwoHundred    = "mul_net_2_100"
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
		return nnet.NewSimpleTrainer(inputDim, outputDim, 1, 50, nnet.Tanh{}, nnet.Linear{})
	case NetTwoFifty:
		return nnet.NewSimpleTrainer(inputDim, outputDim, 2, 50, nnet.Tanh{}, nnet.Linear{})
	case NetThreeFifty:
		return nnet.NewSimpleTrainer(inputDim, outputDim, 3, 50, nnet.Tanh{}, nnet.Linear{})
	case NetTwoHundred:
		return nnet.NewSimpleTrainer(inputDim, outputDim, 2, 100, nnet.Tanh{}, nnet.Linear{})
	case MulNetTwoTwentyFive:
		net, err := nnet.NewSimpleTrainer(inputDim-1, outputDim, 2, 25, nnet.Tanh{}, nnet.Linear{})
		if err != nil {
			return nil, err
		}
		mulnet := mlalg.MulTrainer{net}
		return mulnet, nil
	case MulNetTwoThirtyFive:
		net, err := nnet.NewSimpleTrainer(inputDim-1, outputDim, 2, 35, nnet.Tanh{}, nnet.Linear{})
		if err != nil {
			return nil, err
		}
		mulnet := mlalg.MulTrainer{net}
		return mulnet, nil
	case MulNetTwoFourty:
		net, err := nnet.NewSimpleTrainer(inputDim-1, outputDim, 2, 40, nnet.Tanh{}, nnet.Linear{})
		if err != nil {
			return nil, err
		}
		mulnet := mlalg.MulTrainer{net}
		return mulnet, nil
	case MulNetTwoFourtyFive:
		net, err := nnet.NewSimpleTrainer(inputDim-1, outputDim, 2, 45, nnet.Tanh{}, nnet.Linear{})
		if err != nil {
			return nil, err
		}
		mulnet := mlalg.MulTrainer{net}
		return mulnet, nil
	case MulNetTwoFifty:
		net, err := nnet.NewSimpleTrainer(inputDim-1, outputDim, 2, 50, nnet.Tanh{}, nnet.Linear{})
		if err != nil {
			return nil, err
		}
		mulnet := mlalg.MulTrainer{net}
		return mulnet, nil
	case MulNetThreeFifty:
		net, err := nnet.NewSimpleTrainer(inputDim-1, outputDim, 3, 50, nnet.Tanh{}, nnet.Linear{})
		if err != nil {
			return nil, err
		}
		mulnet := mlalg.MulTrainer{net}
		return mulnet, nil
	case MulNetTwoHundred:
		net, err := nnet.NewSimpleTrainer(inputDim-1, outputDim, 2, 100, nnet.Tanh{}, nnet.Linear{})
		if err != nil {
			return nil, err
		}
		mulnet := mlalg.MulTrainer{net}
		return mulnet, nil
	default:
		return nil, Missing{
			Prefix:  "algorithm setting not found",
			Options: sortedAlgorithm,
		}
	}
}
