import bentoml


# TODO Use this later
def _init_remote_runners(svc: bentoml.Service, debug: bool = False):
    model_runners = {}
    # By default, all the runners will be automatically initialized via .init_local()
    if not debug:
        # Less hardcoded in the future?..
        models_runner_uri = f"tcp://ace-{svc.name.replace('_', '-')}.datascience.svc.cluster.local"
        model_runners = {r.name: models_runner_uri for r in svc.runners}

    # BentoMLContainer.remote_runner_mapping.set(model_runners)
    return model_runners
