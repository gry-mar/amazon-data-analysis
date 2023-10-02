docker build -t amazon .

docker run --rm -p 8888:8888 -v $(pwd):/app -it amazon bash 