package mx

import (
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go-ml.dev/pkg/nn/mx/capi"
	"go-ml.dev/pkg/zorros/zlog"
	"strings"
)

func (identity GraphIdentity) String() string {
	return hex.EncodeToString(identity[:])
}

func (g *Graph) Identity() GraphIdentity {
	if g.identity == nil {
		js := g.ToJson(false)
		g.identity = &GraphIdentity{}
		h := sha1.New()
		h.Write(js)
		bs := h.Sum(nil)
		copy(g.identity[:], bs)
	}
	return *g.identity
}

func (g *Graph) ToJson(withLoss bool) []byte {
	out := g.symLast
	if withLoss {
		out = g.symOut
	}
	return capi.ToJson(out)
}

type GraphJs struct {
	Nodes []struct {
		Op     string
		Name   string
		Attrs  map[string]string
		Inputs []interface{}
	}
}

type SummryArg struct {
	No   int    `yaml:"no""`
	Name string `yaml:"name"`
}

type SummaryRow struct {
	No        int         `yaml:"no"`
	Name      string      `yaml:"name"`
	Operation string      `yaml:"op"`
	Params    int         `yaml:"params"`
	Dim       Dimension   `yaml:"dimension"`
	Args      []SummryArg `yaml:"args"`
}

type Summary []SummaryRow

func (g *Graph) Summary(withLoss bool) Summary {
	var gjs GraphJs
	shapes := g.GetShapes(withLoss)
	js := g.ToJson(withLoss)
	if err := json.Unmarshal(js, &gjs); err != nil {
		panic(err.Error())
	}

	ns := map[string]*SummaryRow{}

	for lyno, ly := range gjs.Nodes {
		if ly.Op != "null" || lyno == 0 {
			n := &SummaryRow{No: len(ns), Name: ly.Name, Operation: ly.Op}
			if len(ly.Inputs) > 0 {
				for _, v := range ly.Inputs {
					inp := v.([]interface{})
					ly2 := gjs.Nodes[int(inp[0].(float64))]
					if ly2.Op == "null" {
						if ly2.Name == "_input" {
							//n.Params += g.Input.Dim().Total()
						} else if ly2.Name == "_label" {
							//n.Params += g.Label.Dim().Total()
						} else if p, ok := g.Params[ly2.Name]; ok {
							n.Params += p.Dim().Total()
						}
					} else {
						n.Args = append(n.Args, SummryArg{ns[ly2.Name].No, ly2.Name})
					}
				}
			}
			if ly.Op == "Activation" {
				n.Operation += "(" + ly.Attrs["act_type"] + ")"
			} else if ly.Op == "SoftmaxActivation" {
				n.Operation += "(" + ly.Attrs["mode"] + ")"
			} else if ly.Op == "Pooling" {
				n.Operation += "(" + ly.Attrs["pool_type"] + ")"
			} else if ly.Op == "Convolution" {
				n.Operation += "(" + ly.Attrs["kernel"] + "/" + ly.Attrs["pad"] + "/" + ly.Attrs["stride"] + ")"
			}

			if dim0, ok := shapes[ly.Name+"_output"]; ok {
				n.Dim = Dim(dim0...)
			} else if dim0, ok := shapes[ly.Name+"_loss"]; ok {
				n.Dim = Dim(dim0...)
			}

			if lyno == 0 {
				n.Dim = g.Input.Dim()
			}

			ns[ly.Name] = n
		}
	}

	r := make(Summary, len(ns))
	for _, v := range ns {
		r[v.No] = *v
	}

	return r
}

func (sry Summary) Print(out func(string)) {
	var nameLen, opLen, parLen, dimLen int = 9, 9, 9, 9
	for _, v := range sry {
		if nameLen < len(v.Name) {
			nameLen = len(v.Name)
		}
		if opLen < len(v.Operation) {
			opLen = len(v.Operation)
		}
		p := fmt.Sprintf("%d", v.Params)
		if parLen < len(p) {
			parLen = len(p)
		}
		p = v.Dim.String()
		if dimLen < len(p) {
			dimLen = len(p)
		}
	}
	fth := fmt.Sprintf("%%-%ds | %%-%ds | %%-%ds | %%%ds", nameLen, opLen, dimLen, parLen)
	ft := fmt.Sprintf("%%-%ds | %%-%ds | %%-%ds | %%%dd", nameLen, opLen, dimLen, parLen)
	npars := 0
	hdr := fmt.Sprintf(fth, "Symbol", "Operation", "Output", "Params #")
	out(hdr)
	out(strings.Repeat("-", len(hdr)))
	for _, v := range sry {
		out(fmt.Sprintf(ft, v.Name, v.Operation, v.Dim, v.Params))
		npars += v.Params
	}
	out(strings.Repeat("-", len(hdr)))
	out(fmt.Sprintf("Total params: %d", npars))
}

func (g *Graph) SummaryOut(withLoss bool, out func(string)) {
	sry := g.Summary(withLoss)
	sry.Print(out)
}

func (g *Graph) LogSummary(withLoss bool) {
	g.SummaryOut(withLoss, func(s string) { zlog.Info(s) })
}

func (g *Graph) PrintSummary(withLoss bool) {
	g.SummaryOut(withLoss, func(s string) { fmt.Println(s) })
}
