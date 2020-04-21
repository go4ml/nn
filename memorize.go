package nn

import (
	"bufio"
	"go-ml.dev/pkg/iokit"
	"go-ml.dev/pkg/base/model"
	"gopkg.in/yaml.v3"
	"io"
)

const ModelPartParams = "params.bin.xz"
const ModelPartSymbol = "symbol.bin.xz"
const ModelPartInfo = "network.yaml"
const ModelPartSummary = "summary.txt"

type mnemosyne struct {
	network  *Network
	features []string
	predicts string
}

func (mm mnemosyne) Memorize(c *model.CollectionWriter) (err error) {
	if err = c.Add(ModelPartInfo, func(wr io.Writer) error {
		en := yaml.NewEncoder(wr)
		return en.Encode(map[string]interface{}{
			"kind": "NN",
			"features": mm.features,
			"predicts": mm.predicts,
		})
	}); err != nil {
		return
	}
	if err = c.AddLzma2(ModelPartParams, func(wr io.Writer) (e error) {
		return mm.network.SaveParams(iokit.Writer(wr))
	}); err != nil {
		return
	}
	if err = c.AddLzma2(ModelPartSymbol, func(wr io.Writer) (e error) {
		return mm.network.SaveSymbol(iokit.Writer(wr))
	}); err != nil {
		return
	}
	if err = c.Add(ModelPartSummary, func(wr io.Writer) (e error) {
		w := bufio.NewWriter(wr)
		mm.network.SummaryOut(false,func(s string){w.WriteString(s+"\n")})
		return nil
	}); err != nil {
		return
	}
	return
}
