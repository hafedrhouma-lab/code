# Data Serving (codename Ace)

## DB conventions

### Tables and functions naming

Tables: `<SERVICE_NAME>_features`
Examples: `vendor_ranking_features`, `vendor_ranking_features_v2`, etc.

Functions: `f_get_<SERVICE_NAME>_features`
Examples: `f_get_vendor_ranking_features`, `f_get_vendor_ranking_features_v2`, etc.

## Local environment setup

Assume that we are on a Mac machine, and Homebrew, Python and Docker Desktop are installed.

### Create Python virtual environment

Preferred way is to use `direnv` and its integration with `pyenv` and Python's `venv`. This way you can work on clean
Python installation and not depend on Homebrew's one (there some questions in the community on how it's configured by
default).

`.envrc` is already present in the repo, so just install `direnv` + `pyenv` + Python version, specified in `.envrc`:

```shell
brew install direnv pyenv just
pyenv install 3.10  # See .envrc for the exact version

# Add hook for direnv for specific shell, check https://direnv.net/docs/hook.html
# For example, for bash:
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc 
source ~/.bashrc 

direnv allow # The local common virtual environment will be created and activated automatically after that

just upgrade-pip
```

`direnv` will automatically activate/deactivate the virtual environment when you enter/leave the project directory.

### Install/upgrade dependencies

We use [`pip-tools`](https://github.com/jazzband/pip-tools) to _lock (freeze)_ packages **per service**, to achieve 
reproducible builds. For that there are a few shell scripts in the repo.

```shell
just nba/upgrade compile # corner case for NBA
just upgrade sync # Update all the requirements (based on requirements*.in files) in the local virtual environment
just compile-release vendor_ranking # Then freeze requirements (requirements.txt) for a specific service
```

### Run

Just run a service (./vendor_ranking/service.py or another) to get the service API running.

You may need to also spin up a local Postgres DB (see `docker-compose.yml` for details).

### Tips & Tricks

Run Docker locally via:

```shell
just refresh-models
just build-bento-release vendor_ranking 
just run-bento-release vendor_ranking
```

### Local testing

In terminal:

```shell
just up-test-env-verbose
just run-test-dev "-v"
```


### CI testing

Since BentoML does not support the use of test server at the moment for integration tests execution.

- We are spinning up an Ace client app (using docker-compose on CI)
- The docker-compose file will create a `postgres` db and an `ace` client app.
- Then in tests we connect to app by firing requests to apis and thus simulate integration tests.

### Linting

[ruff](https://docs.astral.sh/ruff/) is used for linting and formatting.

### API profiling

In case of any performance issues there is a way to find a bottleneck using statistical profiling tools.
One of such tools is [py-spy](https://github.com/benfred/py-spy). It provides results as a flamegraph. </br>
Here is example how to do profiling of the `ultron` service.

```shell
just up-test-env # set infra environment with docker-compose
just run-ultron-api # run Ultron API service
ps -a | grep  "ultron/service.py" | head -n 1 | cut -f1 -d' '  # get the pid of the service
just ultron-api-profiling <PID> "25" "/ultron/v1/items-to-purchase"  # run k6 benchmark along with py-spy
```

### Containerd instead of docker runtime
Install `lima`:
```shell
brew install lima
```
Extend `mounts:` section in lima's config `~/.lima/default/lima.yaml`:
```yaml
  - location: "/tmp/data"
    writable: true
```
Start lima:
```shell
lima start
```
Set env var `USE_LIMA_CONTAINERD=1`. Use `lima nerdctl compose` instead of `docker compose`.


### MLFlow
#### Run MLFlow server
1. Run local MLFlow server `just mlflow-up`
2. Access MLFlow CLI tool with `just mlflowcli`  
3. Access MinIO UI with http://localhost:9003 with login/password = s3_access_key/s3_secret_key
4. Access MLFlow UI with http://0.0.0.0:5000
5. Access Jupiter notebook server with http://127.0.0.1:8888
#### Training example
```bash
just mlflowcli run "https://github.com/mlflow/mlflow-example.git -P alpha=0.42"
```
#### Troubleshooting
Restart lima
```bash
limactl stop default
limactl start default
```
