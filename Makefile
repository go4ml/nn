build:
	go build ./...

run-tests:
	cd tests && go test -o ../tests.test -c -covermode=atomic -coverprofile=c.out -coverpkg=../...
	./tests.test -test.v=true -test.coverprofile=c.out
	cp c.out gocov.txt
	sed -i -e 's:go-ml.dev/pkg/nn/::g' c.out

run-cover:
	go tool cover -html=gocov.txt

run-cover-tests: run-tests run-cover

