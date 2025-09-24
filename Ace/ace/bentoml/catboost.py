import os
from typing import Type, Any, Callable

import bentoml
import catboost as cb
import numpy as np
from bentoml.exceptions import InvalidArgument


def get_runnable(bento_model: bentoml.Model) -> Type[bentoml.Runnable]:
    from bentoml._internal.frameworks.catboost import load_model
    from bentoml._internal.models.model import ModelSignature

    class CatBoostRunnable(bentoml.Runnable):
        """
        'task_type' is not supported by CatBoostRanker.predict() (but supported by evert other method), the only way to
        remove it is this one...
        """

        SUPPORTED_RESOURCES = "cpu"
        SUPPORTS_CPU_MULTI_THREADING = True

        predict_params: dict[str, Any] = {}

        def __init__(self):
            super().__init__()
            self.model = load_model(bento_model)

            # check for resources
            nthreads = os.getenv("OMP_NUM_THREADS")
            if nthreads is not None and nthreads != "":
                nthreads = max(int(nthreads), 1)
            else:
                nthreads = -1
            self.predict_params["thread_count"] = nthreads

            self.predict_fns: dict[str, Callable[..., Any]] = {}
            for method_name in bento_model.info.signatures:
                try:
                    self.predict_fns[method_name] = getattr(self.model, method_name)
                except AttributeError:
                    raise InvalidArgument(
                        f"No method with name {method_name} found for CatBoost model of type {self.model.__class__}"
                    )

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(
            self: CatBoostRunnable,
            input_data,
        ):
            if not isinstance(input_data, cb.Pool):
                input_data = cb.Pool(input_data)
            res = self.predict_fns[method_name](input_data, **self.predict_params)
            return np.asarray(res)

        CatBoostRunnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in bento_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return CatBoostRunnable
