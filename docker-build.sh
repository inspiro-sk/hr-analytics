#!/bin/bash
imageName=rstana/api_serve:latest
containerName=api_serve

docker build -t $imageName -f Dockerfile  .

echo Delete old container...
docker rm -f $containerName

echo Run new container...
docker run -d -p 50000:5000 --name $containerName $imageName