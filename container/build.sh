#!/bin/bash

# Build the Singularity/Apptainer image
# Using --fakeroot since we don't have sudo on the cluster
singularity build --fakeroot alchemi.sif alchemi.def
