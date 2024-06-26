#!/bin/bash

# List of required Python packages
required_packages=("numpy" "pandas" "matplotlib" "random" "gymnasium" "typing" "collections" "tqdm" "itertools" "collections")

# Check if each package is installed
for package in "${required_packages[@]}"; do
    if ! python -c "import $package" &> /dev/null; then
        echo "Installing $package..."
        conda install $package
    else
        echo "$package is already installed."
    fi
done