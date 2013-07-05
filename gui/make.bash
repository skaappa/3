#! /bin/bash
echo "package gui" > js.go
echo "// THIS FILE IS AUTO-GENERATED BY make.bash" >> js.go
echo "// THIS FILE IS AUTO-GENERATED BY make.bash" >> js.go
echo "" >> js.go
echo "const js = \`<script type=\"text/javascript\">" >> js.go
cat script.js >> js.go
echo "</script>\`" >> js.go

go build -v
go build test.go
