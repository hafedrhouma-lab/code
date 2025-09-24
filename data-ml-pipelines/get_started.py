import os
import shutil
import sys
import logging

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class GetStarted:

    def __init__(self):
        self.TEMPLATE_PATH = 'template'
        self.TARGET_PATH = 'projects'
        self.project_name = None
        self.model_name = None
        self.project_path = None
        self.model_path = None

    def create_new_project(self):
        self.project_path = os.path.join(self.TARGET_PATH, self.project_name)
        if os.path.exists(self.project_path):
            raise FileExistsError(
                f"Project '{self.project_name}' already exists. Please choose another project name or choose to create a model under the existing project.")
        os.makedirs(self.project_path)
        # creating a "common" folder for all projects
        common_path = os.path.join(self.project_path, "common")
        if not os.path.exists(common_path):
            os.makedirs(common_path)
        # create a "__init__.py" file in the "common" folder
        with open(os.path.join(common_path, "__init__.py"), "w") as file:
            pass

    def create_new_model(self):
        project_path = os.path.join(self.TARGET_PATH, self.project_name)
        self.model_path = os.path.join(project_path, self.model_name)

        if os.path.exists(self.model_path):
            print(f"Model '{self.model_name}' already exists in project '{self.project_name}'. Please choose another model name.")
            sys.exit(0)

        shutil.copytree(self.TEMPLATE_PATH, self.model_path)
        self.replace_model_name_in_file()
        LOG.info(
            f"Creating model '{self.model_name}' in project '{self.project_name}' at {self.model_path}")


    def replace_model_name_in_file(self):
        for filename in os.listdir(self.model_path):
            file_path = os.path.join(self.model_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    content = file.read()

                new_content = content.replace("<project_name>", f'{self.project_name}')
                new_content = new_content.replace("<model_name>", f'{self.model_name}')

                with open(file_path, 'w') as file:
                    file.write(new_content)

    def __call__(self):
        # Prompt the user for the project name
        self.project_name = input(
            "Enter the name of the project: ").strip()

        self.project_path = os.path.join(self.TARGET_PATH, self.project_name)

        # Check if the project exists
        if os.path.exists(self.project_path):
            # Prompt the user if they want to create a new model in the existing project
            choice = input(
                f"Project '{self.project_name}' already exists. Do you want to create a new model in this project? (yes/no): ").strip().lower()
            if choice == "yes":
                self.model_name = input(
                    "Enter the name of the new model: ").strip()
                self.create_new_model()
            else:
                print("Operation cancelled. No new model was created.")
                sys.exit(0)

        else:
            # Prompt the user if they are sure they want to create a new project
            choice = input(
                f"Project '{self.project_name}' does not exist. Do you want to create this project? (yes/no): ").strip().lower()
            if choice == "yes":
                self.create_new_project()
                self.model_name = input(
                    "Enter the name of the first model: ").strip()
                self.create_new_model()
            else:
                print("Operation cancelled. No new project was created.")
                sys.exit(0)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        print(
            "This script doesn't take any arguments, just run 'python get_started.py' in your terminal.")
        sys.exit(1)

    get_started = GetStarted()
    get_started()
