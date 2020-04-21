package nn

import (
	"go-ml.dev/pkg/iokit"
	"go-ml.dev/pkg/base/fu"
	"go-ml.dev/pkg/base/model"
	"go-ml.dev/pkg/nn/mx"
	"go-ml.dev/pkg/base/tables"
	"go-ml.dev/pkg/zorros"
	"golang.org/x/xerrors"
	"gopkg.in/yaml.v3"
	"io"
)

// default batch size for general nn training
const DefaultBatchSize = 32

/*
Model is a ANN/Gym definition to train network
*/
type Model struct {
	Network   Block
	Optimizer OptimizerConf
	Loss      mx.Loss
	Input     mx.Dimension
	Seed      int
	BatchSize int
	Predicted string
	Context   mx.Context // CPU by default
}

func (e Model) Feed(ds model.Dataset) model.FatModel {
	return func(workout model.Workout) (*model.Report, error) {
		return Train(e, ds, workout, DefaultModelMap)
	}
}

/*
PredictionModel is the FeaturesMapper factory
*/
type PredictionModel struct {
	features       []string
	predicts       string
	symbol, params iokit.Input
	context        mx.Context
}

/*
Features model uses when maps features
the same as Features in the training dataset
*/
func (pm PredictionModel) Features() []string { return pm.features }

/*
Column name model adds to result table when maps features.
By default it's 'Predicted'
*/
func (pm PredictionModel) Predicted() string { return pm.predicts }

/*
Returns new table with all original columns except features
adding one new column with prediction
*/
func (pm PredictionModel) FeaturesMapper(batchSize int) (fm tables.FeaturesMapper, err error) {
	network, err := Load(pm.context, pm.symbol, pm.params, batchSize)
	if err != nil {
		return
	}
	fm = &FeaturesMapper{model: pm, network: network}
	return
}

/*
Gpu changes context of prediction network to gpu enabled
*/
func (pm PredictionModel) Gpu(no ...int) model.PredictionModel {
	pm.context = mx.GPU0
	if len(no) > 0 {
		pm.context = mx.Gpu(no[0])
	}
	return pm
}

/*
FeaturesMapper maps features to prediction
*/
type FeaturesMapper struct {
	model   PredictionModel
	network *Network
}

/*
MapFeature returns new table with all original columns except features
adding one new column with prediction/calculation
*/
func (fm *FeaturesMapper) MapFeatures(t *tables.Table) (r *tables.Table, err error) {
	var input tables.Matrix
	if input, err = t.Matrix(fm.model.features, fm.network.BatchSize); err != nil {
		return
	}
	out := make([]float32, fm.network.Output.Dim().Total())
	outWidth := fm.network.Output.Dim().Total() / fm.network.BatchSize
	if input.Width != fm.network.Input.Dim().Total()/fm.network.BatchSize {
		return nil, xerrors.Errorf("features does not fit network input")
	}
	if t.Len() > fm.network.BatchSize {
		return nil, xerrors.Errorf("batch size does not fit network input")
	}
	fm.network.Forward(input.Features, out)
	return t.Except(fm.model.features...).With(tables.MatrixColumn(out[0:outWidth*t.Len()], t.Len()), fm.model.predicts), nil
}

/*
Close releases all bounded resources
*/
func (fm *FeaturesMapper) Close() error {
	fm.network.Release()
	return nil
}

func ObjectifyModel(c map[string]iokit.Input) (pm model.PredictionModel, err error) {
	var rd io.ReadCloser
	if _, ok := c[ModelPartInfo]; !ok {
		return nil,zorros.New("it's not neural network model")
	}
	if rd, err = c[ModelPartInfo].Open(); err != nil {
		return
	}
	defer rd.Close()
	cf := map[string]interface{}{}
	if err = yaml.NewDecoder(rd).Decode(&cf); err != nil {
		return
	}
	m := PredictionModel{
		symbol:   c[ModelPartSymbol],
		params:   c[ModelPartParams],
		features: fu.Strings(cf["features"]),
		predicts: cf["predicts"].(string),
	}
	return m, nil
}

func Objectify(source iokit.Input, collection ...string) (fm model.GpuPredictionModel, err error) {
	x := fu.Fnzs(fu.Fnzs(collection...), "model")
	m, err := model.Objectify(source, model.ObjectifyMap{x: ObjectifyModel})
	if err != nil {
		return
	}
	return m[x].(model.GpuPredictionModel), nil
}

func LuckyObjectify(source iokit.Input, collection ...string) model.GpuPredictionModel {
	fm, err := Objectify(source, collection...)
	if err != nil {
		panic(err)
	}
	return fm
}
