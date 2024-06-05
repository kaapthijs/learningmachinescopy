# We base ourselves on Unbuntu. This is the base OS we are installing.
FROM ubuntu:20.04

# Let's CD to a more sensible working directory
WORKDIR /root/workdir

# Copy the file from your own machine to the container
COPY ./README.md ./README.md

# Specify what to run, in this case, the `head` command (print the first few lines of a file to the terminal) with as argument `./README.md`
CMD ["head", "./README.md"]