package nn

import (
	"go4ml.xyz/base/fu"
	"go4ml.xyz/base/model"
	"go4ml.xyz/base/tables"
	"go4ml.xyz/zorros"
	"reflect"
)

type ModelMapFunction func(network *Network, features []string, predicts string) model.MemorizeMap

func DefaultModelMap(network *Network, features []string, predicts string) model.MemorizeMap {
	return model.MemorizeMap{"model": mnemosyne{network, features, predicts}}
}

func Train(e Model, dataset model.Dataset, w model.Workout, mmf ModelMapFunction) (report *model.Report, err error) {
	t, err := dataset.Source.Lazy().First(1).Collect()
	if err != nil {
		return
	}

	features := t.OnlyNames(dataset.Features...)

	Test := fu.Fnzs(dataset.Test, model.TestCol)
	if fu.IndexOf(Test, t.Names()) < 0 {
		err = zorros.Errorf("dataset does not have column `%v`", Test)
		return
	}

	Label := fu.Fnzs(dataset.Label, model.LabelCol)
	if fu.IndexOf(Label, t.Names()) < 0 {
		err = zorros.Errorf("dataset does not have column `%v`", Label)
		return
	}

	if e.BatchSize <= 0 {
		e.BatchSize = DefaultBatchSize
	}

	predicts := fu.Fnzs(e.Predicted, model.PredictedCol)

	network := New(e.Context.Upgrade(), e.Network, e.Input, e.Loss, e.BatchSize, e.Seed)
	train := dataset.Source.Lazy().IfNotFlag(dataset.Test).Batch(e.BatchSize).Parallel()
	full := dataset.Source.Lazy().Batch(e.BatchSize).Parallel()
	out := make([]float32, network.Graph.Output.Dim().Total())
	loss := make([]float32, network.Graph.Loss.Dim().Total())

	network.SummaryOut(true, w.Verbose)

	for done := false; w != nil && !done; w = w.Next() {
		opt := e.Optimizer.Init(w.Iteration())

		if err = train.Drain(func(value reflect.Value) error {
			if value.Kind() == reflect.Bool {
				return nil
			}
			t := value.Interface().(*tables.Table)
			m, err := t.MatrixWithLabel(features, Label, e.BatchSize)
			if err != nil {
				return err
			}
			network.Train(m.Features, m.Labels, opt)
			return nil
		}); err != nil {
			return
		}

		trainmu := w.TrainMetrics()
		testmu := w.TestMetrics()
		if err = full.Drain(func(value reflect.Value) error {
			if value.Kind() == reflect.Bool {
				return nil
			}
			t := value.Interface().(*tables.Table)
			m, err := t.MatrixWithLabel(features, Label, e.BatchSize)
			if err != nil {
				return err
			}
			network.Label.SetValues(m.Labels)
			network.Forward(m.Features, out)
			resultCol := tables.MatrixColumn(out, e.BatchSize)
			labelCol := t.Col(Label)
			network.Loss.CopyValuesTo(loss)

			l := loss[0]
			for i, c := range t.Col(Test).ExtractAs(fu.Bool, true).([]bool) {
				if len(loss) > 1 {
					l = loss[i]
				}
				if c {
					testmu.Update(resultCol.Value(i), labelCol.Value(i), float64(l))
				} else {
					trainmu.Update(resultCol.Value(i), labelCol.Value(i), float64(l))
				}
			}
			return nil
		}); err != nil {
			return
		}

		lr0, _ := trainmu.Complete()
		lr1, d := testmu.Complete()
		memorize := mmf(network, features, predicts)
		if report, done, err = w.Complete(memorize, lr0, lr1, d); err != nil {
			return nil, zorros.Wrapf(err, "tailed to complete model: %s", err.Error())
		}
	}

	return
}
