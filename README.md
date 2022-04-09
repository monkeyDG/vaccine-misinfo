You can build and run the docker image using:
build -t vaccine-misinfo:1.0 . 
docker run -p80:80 -p5000:5000 vaccine-misinfo:1.0