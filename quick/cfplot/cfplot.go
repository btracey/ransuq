package main

import (
	"flag"
	"path/filepath"

	"code.google.com/p/gocircuit/src/circuit/load/config"

	"github.com/btracey/ransuq/datawrapper"
	"gituhb.com/btracey/su2tools/driver"
)

func main() {
	var truename string
	flag.StringVar(*truename, "true", "", "surface flow file of the true")
	var mlname string
	flag.StringVar(*truename, "ml", "", "surface flow file of the ml version")

	mldir := filepath.Dir(mlname)
	mlsol := filepath.Base(mlname)
	mlconfig := &config.Config{}

	origdir := filepath.Dir(truename)

	mlcompdir := filepath.Join(mldir, "postprocess")

	su2 := &datawrapper.SU2{
		Driver: &driver.Driver{
			Wd: mldir,
		},
	}

	su2ml := &datawrapper.SU2ML{
		PostprocessDir: mlcompdir,
	}

	fp := datawrapper.FlatplatePostprocessor{}
}
