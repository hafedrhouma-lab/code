import pandas as pd
from pandas import json_normalize
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.test_suite import TestSuite
from evidently.tests import TestRecallTopK


def target_drift_test(reference_data, current_data):
    """Statistical Test to detect drift in Recall@10"""
    column_mapping = ColumnMapping(
        recommendations_type='rank',
        target='target',
        prediction='prediction',
        item_id='item_id',
        user_id='user_id'
    )

    tests_target_drift = TestSuite(tests=[
        TestRecallTopK(k=9, no_feedback_users=True),
    ])

    tests_target_drift.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    target_drift = tests_target_drift.as_dict()
    target_drift_test_df = pd.DataFrame(target_drift['tests'])

    parameters_df = json_normalize(
        target_drift_test_df['parameters']
    )
    target_drift_df = pd.concat(
        [
            target_drift_test_df.drop(columns=["parameters"]),
            parameters_df
        ],
        axis=1
    )
    target_drift_df = target_drift_df.drop(
        columns=["condition.eq.absolute", "group"]
    )
    target_drift_df = target_drift_df.rename(columns={
        'condition.eq.value': 'condition_eq_value',
        'condition.eq.relative': 'condition_eq_relative'
    })

    return target_drift_df
