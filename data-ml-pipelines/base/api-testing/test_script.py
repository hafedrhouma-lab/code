import os
import json
import yaml

from pathlib import Path
from pydantic import ValidationError

from config import ApiTestingConfig

TESTS_DIR_NAME = "api-tests"
BASE_PATH = Path(__file__).resolve().parent
PROJECTS_DIR = BASE_PATH.parents[1] / 'projects'
OUTPUT_FILE = f"{BASE_PATH}/collections.json"


def parse_config(config_path):
    """
    Parses and validates the config.yaml file using Pydantic schema.

    Args:
        config_path (Path): Path to the config.yaml file.

    Returns:
        dict: Parsed and validated configuration.

    Raises:
        ValidationError: If the config file does not conform to the schema.
    """
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    try:
        validated_config = ApiTestingConfig(**config_data)
        return validated_config.model_dump()
    except ValidationError as e:
        print(f"Validation error in {config_path}: {e}")
        raise


def collect_tests(base_dir):
    """
    Traverses the directory structure to find api-tests directories and collects test configurations.

    Args:
        base_dir (Path): Base directory to traverse.

    Returns:
        list: List of test cases containing validated configurations.
    """
    test_cases = []

    print(f"Starting directory traversal from: {base_dir}")
    for root, dirs, _ in os.walk(base_dir):
        if TESTS_DIR_NAME in dirs:
            api_testing_dir = Path(root) / TESTS_DIR_NAME
            print(f"Found {TESTS_DIR_NAME} directory in: {root}")

            for test_dir in api_testing_dir.iterdir():
                if test_dir.is_dir():
                    config_file = test_dir / "config.yaml"
                    req_file = test_dir / "req.json"
                    res_file = test_dir / "res.json"

                    if not (config_file.exists() and req_file.exists() and res_file.exists()):
                        print(f"Skipping {test_dir}: Missing required files.")
                        continue

                    try:
                        config = parse_config(config_file)
                        with open(req_file, "r") as req, open(res_file, "r") as res:
                            request_data = json.load(req)
                            response_data = json.load(res)

                        test_cases.append({
                            "name": config["setup"]["name"],
                            "url": config["setup"]["url"],
                            "method": config["setup"]["method"],
                            "assertions": config["setup"]["assertions"],
                            "request": request_data,
                            "expected_response": response_data
                        })
                        print(f"Successfully validated and added test case from: {test_dir}")
                    except (ValidationError, json.JSONDecodeError) as e:
                        print(f"Error processing {test_dir}: {e}")

    return test_cases


def generate_postman_collection(test_cases, output_file):
    """
    Generates a Postman collection JSON file from the test cases.

    Args:
        test_cases (list): List of test cases with validated configurations.
        output_file (Path): Path to save the Postman collection JSON file.
    """
    collection = {
        "info": {
            "name": "API Testing Collection",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "item": []
    }

    for test in test_cases:
        test_item = {
            "name": test["name"],
            "request": {
                "method": test["method"],
                "header": [
                    {"key": "User-Agent", "value": "{{USER_AGENT}}"},
                    {"key": "Content-Type", "value": "application/json"}
                ],
                "url": str(test["url"]),
                "body": {
                    "mode": "raw",
                    "raw": json.dumps(test["request"], indent=4)
                }
            },
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            f"pm.test(\"Status code is {test['assertions']['status_code']}\", function () {{",
                            f"    pm.response.to.have.status({test['assertions']['status_code']});",
                            "});",

                            f"pm.test(\"Validate response\", function () {{",
                            f" const expectedResponse={test['expected_response']}; const actualResponse = pm.response.json(); pm.expect(actualResponse).to.deep.equal(expectedResponse);",
                            "});",
                        ]
                    }
                }
            ]
        }

        if "optional" in test["assertions"]:
            optional = test["assertions"]["optional"]
            if optional.get("headers"):
                for header in optional["headers"]:
                    test_item["event"][0]["script"]["exec"].append(
                        f"pm.test(\"Header {header[0]} is {header[1]}\", function () {{\n    pm.expect(pm.response.headers.get(\"{header[0]}\")).to.eql(\"{header[1]}\");\n}});"
                    )

        collection["item"].append(test_item)

    with open(output_file, "w") as file:
        json.dump(collection, file, indent=4)


if __name__ == "__main__":
    base_directory = Path(PROJECTS_DIR)
    output_path = Path(OUTPUT_FILE)

    test_cases = collect_tests(base_directory)
    generate_postman_collection(test_cases, output_path)
    print(f"Postman collection generated at {output_path}")
