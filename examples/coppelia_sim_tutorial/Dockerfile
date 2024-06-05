# We base ourselves on Unbuntu. This is the base OS we are installing.
FROM ubuntu:20.04

# Run apt-get to install git
RUN apt-get -y update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Specify what to run, in this case, the `echo` command with as argument `Hello world`
CMD ["echo", "Hello World"]