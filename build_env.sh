#!/bin/bash

echo "Select an option:"
echo "1) Build a new environment"
echo "2) Enter an existing environment"
echo "3) Exit the current environment"
read -p "Enter your choice (1/2/3): " choice

case $choice in
    1)
        echo "Checking if the environment already exists..."
        if conda env list | grep -q 'analytics_env'; then
            echo "Environment 'analytics_env' already exists. Removing it..."
            conda env remove -n analytics_env
        fi
        echo "Creating a new environment..."
        if conda env create -f environment.yaml; then
            echo "Environment created."
        else
            echo "Failed to create the environment."
            exit 1
        fi
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
