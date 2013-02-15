// +build ignore

/*
 This program generates Go wrappers for cuda sources.
 The cuda file should contain exactly one __global__ void.
*/
package main

import (
	"bufio"
	"bytes"
	"code.google.com/p/mx3/util"
	"flag"
	"fmt"
	"io"
	"os"
	"text/scanner"
	"text/template"
)

func main() {
	flag.Parse()
	for _, fname := range flag.Args() {
		cuda2go(fname)
	}
}

// generate cuda wrapper for file.
func cuda2go(fname string) {
	// open cuda file
	f, err := os.Open(fname)
	util.PanicErr(err)
	defer f.Close()

	// read tokens
	var token []string
	var s scanner.Scanner
	s.Init(f)
	tok := s.Scan()
	for tok != scanner.EOF {
		if !filter(s.TokenText()) {
			token = append(token, s.TokenText())
		}
		tok = s.Scan()
	}

	// find function name and arguments
	funcname := ""
	argstart, argstop := -1, -1
	for i := 0; i < len(token); i++ {
		if token[i] == "__global__" {
			funcname = token[i+2]
			argstart = i + 4
		}
		if argstart > 0 && token[i] == ")" {
			argstop = i + 1
			break
		}
	}
	argl := token[argstart:argstop]

	// isolate individual arguments
	var args [][]string
	start := 0
	for i, a := range argl {
		if a == "," || a == ")" {
			args = append(args, argl[start:i])
			start = i + 1
		}
	}

	// separate arg names/types and make pointers Go-style
	argn := make([]string, len(args))
	argt := make([]string, len(args))
	for i := range args {
		if args[i][1] == "*" {
			args[i] = []string{args[i][0] + "*", args[i][2]}
		}
		argt[i] = typemap(args[i][0])
		argn[i] = args[i][1]
	}
	wrapgen(fname, funcname, argt, argn)
}

var tm = map[string]string{"float*": "unsafe.Pointer", "float": "float32", "int": "int"}

// translate C type to Go type.
func typemap(ctype string) string {
	if gotype, ok := tm[ctype]; ok {
		return gotype
	}
	panic(fmt.Errorf("unsupported cuda type: %v", ctype))
	return "" // unreachable
}

// template data
type Kernel struct {
	Name string
	ArgT []string
	ArgN []string
	PTX  string
}

// generate wrapper code from template
func wrapgen(filename, funcname string, argt, argn []string) {
	ptx := filterptx(util.NoExt(filename) + ".ptx")
	kernel := &Kernel{funcname, argt, argn, "`" + string(ptx) + "`"}
	wrapfname := util.NoExt(filename) + ".go"
	wrapout, err := os.OpenFile(wrapfname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	util.PanicErr(err)
	defer wrapout.Close()
	util.PanicErr(templ.Execute(wrapout, kernel))
}

// wrapper code template text
const templText = `package kernel

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import(
	"unsafe"
	"github.com/barnex/cuda5/cu"
)

var {{.Name}}_code cu.Function

type {{.Name}}_args struct{
	{{range $i, $_ := .ArgN}} arg_{{.}} {{index $.ArgT $i}}
	{{end}} argptr [{{len .ArgN}}]unsafe.Pointer
}

// Wrapper for {{.Name}} CUDA kernel. Synchronizes before return.
func K_{{.Name}} ( {{range $i, $t := .ArgT}}{{index $.ArgN $i}} {{$t}}, {{end}} gridDim, blockDim cu.Dim3) {
	if {{.Name}}_code == 0{
		{{.Name}}_code = cu.ModuleLoadData({{.Name}}_ptx).GetFunction("{{.Name}}")
	}

	var a {{.Name}}_args

	{{range $i, $t := .ArgN}} a.arg_{{.}} = {{.}}
	a.argptr[{{$i}}] = unsafe.Pointer(&a.arg_{{.}})
	{{end}}

	args := a.argptr[:]
	str := Stream()
	cu.LaunchKernel({{.Name}}_code, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, 0, str, args)
	SyncAndRecycle(str)
}

const {{.Name}}_ptx = {{.PTX}} `

// wrapper code template
var templ = template.Must(template.New("wrap").Parse(templText))

// should token be filtered out of stream?
func filter(token string) bool {
	switch token {
	case "__restrict__":
		return true
	}
	return false
}

// Filter comments and ".file" entries from ptx code.
// They spoil the git history
func filterptx(fname string) string {
	f, err := os.Open(fname)
	util.PanicErr(err)
	defer f.Close()
	in := bufio.NewReader(f)
	var out bytes.Buffer
	line, err := in.ReadBytes('\n')
	for err != io.EOF {
		util.PanicErr(err)
		if !bytes.HasPrefix(line, []byte("//")) && !bytes.HasPrefix(line, []byte("	.file")) {
			out.Write(line)
		}
		line, err = in.ReadBytes('\n')
	}
	return out.String()
}
