#!/bin/bash

# Set the root directory for the projects
PROJECTS_DIR="projects"

# Check if the projects directory exists
if [ ! -d "$PROJECTS_DIR" ]; then
  echo "Error: Directory '$PROJECTS_DIR' does not exist."
  exit 1
fi

# Find all api-tests directories
echo "Searching for api-tests directories in '$PROJECTS_DIR'..."
API_TESTS_DIRS=$(find "$PROJECTS_DIR" -type d -name "api-tests")

if [ -z "$API_TESTS_DIRS" ]; then
  echo "No api-tests directories found."
  exit 0
fi

# Loop through each api-tests directory
for DIR in $API_TESTS_DIRS; do
  COLLECTION_FILE="$DIR/collection.json"

  # Check if the collection.json file exists
  if [ -f "$COLLECTION_FILE" ]; then
    echo "Found collection file: $COLLECTION_FILE"

    # Run newman on the collection file
    newman run "$COLLECTION_FILE"
    if [ $? -ne 0 ]; then
      echo "Newman run failed for collection: $COLLECTION_FILE"
    else
      echo "Newman run succeeded for collection: $COLLECTION_FILE"
    fi
  else
    echo "No collection.json file found in: $DIR"
  fi
done