import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from ..utils.eval_utils import (
    get_embeddings_similarity,
    weighted_recall,
    list_intersection_count,
    remove_redundant_values
)


def create_guest_sample(data):
    x = data.sample(1)
    x["prev_clicks"] = "first_click"
    x["prev_searches"] = "no_prev_search"
    x["user_prev_chains"] = "first_order"
    if "freq_clicks" in data.columns:
        x["freq_clicks"] = "no_frequent_clicks"
        x["freq_chains"] = "no_frequent_orders"
    x["account_id"] = "-1"
    return x


class OrderLevelRecallatK(tf.keras.callbacks.Callback):
    def __init__(
            self,
            k,
            geo_to_chains,
            geo_to_parent_geo,
            test_df,
            train_sample_df,
            query_features,
            candidate_features,
            account_prev_interactions,
            chain_features_df,
            chain_names=None,
            order_level_eval=True,
            logger=None,
            best_first_epoch_recall=None

    ):
        self.k = k

        self.geo_to_chains = geo_to_chains
        self.geo_to_parent_geo = geo_to_parent_geo

        self.test_df = test_df
        self.train_sample_df = train_sample_df
        self.query_features = query_features
        self.candidate_features = candidate_features
        self.account_prev_interactions = account_prev_interactions
        self.order_level_eval = order_level_eval
        self.chains_features_df = chain_features_df
        self.recall_list = {}
        self.chain_to_name = {
            str(a): b for a, b in zip(chain_names.chain_id.values, chain_names.chain_name.values)
        }

        self.best_recall = best_first_epoch_recall
        self.logger = logger

    def on_train_begin(self, logs=None):
        print(f"TRAINING STARTED | Dollar is falling\n")

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 3 == 0 or epoch in range(15):
            self.logger.info(f"Epoch {epoch + 1}\n\n\n")
            if self.order_level_eval:
                self.compute_embeddings_dynamic_features()
                self.evaluate_tower_on_orders(
                    epoch=epoch,
                    account_prev_interactions=self.account_prev_interactions,
                    data=self.test_df,
                    evaluation_str="test"
                )

            else:
                print("Tower Eval @Epoch", epoch)
                self.compute_embeddings_dynamic_features()
                self.evaluate_tower_on_sample(epoch=epoch)


    def compute_embeddings_dynamic_features(self):
        first_order_df = self.test_df
        first_order_df["rank"] = (
                first_order_df.sort_values(
                    ["order_time_utc", "account_id"], ascending=[True, False]
                )
                .groupby(["account_id"])
                .cumcount()
                + 1
        )

        first_order_df = first_order_df.query("rank == 1")

        assert first_order_df.account_id.value_counts().max() == 1

        c_model = self.model.chain_model

        # unique_chains = self.test_df[self.candidate_features].drop_duplicates()
        # FIXED this for feature store
        chains_df = self.test_df.sort_values(by=['chain_id', 'order_date'], ascending=[True, False])
        unique_chains = chains_df.drop_duplicates(subset='chain_id', keep='first')
        unique_chains = unique_chains[self.candidate_features]
        # FIXED this for feature store

        print("unique chains from test set", unique_chains.chain_id.nunique())
        print("unique chains from train set", self.chains_features_df.chain_id.nunique())
        chain_embeddings_np = c_model(unique_chains).numpy()
        chain_embeddings = pd.DataFrame(
            data={
                "chain_id": unique_chains.chain_id,
                "chain_embeddings": list(chain_embeddings_np),
            }
        )
        self.chain_embeddings = chain_embeddings.set_index("chain_id")

    def evaluate_tower_on_orders(self, epoch, account_prev_interactions, data, evaluation_str):
        reorder_count = 0
        new_order_count = 0
        recall_dict = {}
        orders_count = 0
        exceptions = 0
        not_found_errors = 0
        total_displayed = 0
        customer_model = self.model.customer_model

        sample_test_orders = data

        x = create_guest_sample(data)
        sample_test_orders = pd.concat(
            [sample_test_orders, self.test_df[self.test_df.account_id == "27126109"], x],
            axis=0,
            ignore_index=True
        )
        print(sample_test_orders.iloc[-5:])

        model_features = sample_test_orders[self.query_features]
        coverage = {}
        sample_query_embeddings = customer_model(model_features).numpy()
        print(
            "evaluating on ",
            sample_test_orders.shape[0] / self.test_df.shape[0],
            f"% of {evaluation_str} data",
        )
        if self.logger:
            self.logger.info(
                "evaluating on " +
                str(sample_test_orders.shape[0] / self.test_df.shape[0]) +
                f"% of {evaluation_str} data"
            )
        for i, order in tqdm(
                enumerate(sample_test_orders.itertuples()),
                total=sample_test_orders.shape[0],
        ):
            try:
                geohash = order.geohash
                chain_id = order.chain_id
                account_id = order.account_id
                chain_name = order.chain_name
                user_prev_chains = order.user_prev_chains
                if "freq_chains" in self.query_features and account_id != -1:
                    prev_chains = order.freq_chains + " " + order.user_prev_chains
                else:
                    prev_chains = user_prev_chains
                query_embeddings = sample_query_embeddings[i]

                # TODO this supposed to be
                # available_chains = list(
                #     self.geo_to_chains[self.geo_to_parent_geo[geohash]]
                # )
                available_chains_w_embedding = [
                    str(x)
                    # for x in available_chains
                    for x in self.chain_embeddings.index
                    # if str(x) in self.chain_embeddings.index
                ]

                # TODO: handle the following line better
                if chain_id not in available_chains_w_embedding:
                    not_found_errors += 1
                    continue

                chains_embeddings = self.chain_embeddings.loc[
                    available_chains_w_embedding
                ].values
                chains_embeddings = np.array(
                    [np.array([x for x in y]) for y in chains_embeddings]
                ).squeeze()
                account_embeddings = query_embeddings

                sim_score = get_embeddings_similarity(
                    account_embeddings, chains_embeddings
                )

                chain_order = np.argsort(sim_score)[::-1]
                assert len(chain_order) == len(available_chains_w_embedding)
                sorted_chains = np.array(available_chains_w_embedding)[chain_order]
                # print("sorted_chains: ", sorted_chains[0:2])
                unique_sorted_chains = [
                    self.chain_to_name[chain_id] for chain_id in sorted_chains
                ]
                unique_sorted_chains = remove_redundant_values(unique_sorted_chains)
                current_order_rnk = np.where(sorted_chains == chain_id)[0][0]

                prev_chain_ids = [int(x) for x in user_prev_chains.split() if len(user_prev_chains.split()) > 1]
                prev_order_rnks = [np.where(sorted_chains == str(ch_id))[0][0]
                                   if str(ch_id) in available_chains_w_embedding else -1
                                   for ch_id in prev_chain_ids]

                if (
                        np.random.rand() < 0.01 and total_displayed <= 5
                ) or account_id in ["27126109", "-1"]:
                    top10_recommendations = np.array(available_chains_w_embedding)[
                        chain_order[:25]
                    ]
                    for i in range(20):
                        coverage[i + 1] = coverage.get(i + 1, set()).intersection(
                            top10_recommendations[i]
                        )
                    if account_id == "27126109":
                        print("Omar Recommendations")
                        if self.logger:
                            self.logger.info("Omar Recommendations")
                    if account_id == "-1":
                        print("Guest Recommendations")
                        if self.logger:
                            self.logger.info(" Guest Recommendations")
                    self.display_recommendation(order, top10_recommendations,
                                                order_rnk=current_order_rnk,
                                                prev_order_rnks=prev_order_rnks)
                    total_displayed += 1
                for k in self.k:
                    retrieved_chains = np.array(available_chains_w_embedding)[
                        chain_order[:k]
                    ]
                    retrieved_names = np.array(unique_sorted_chains)[:k]
                    recall_k = 1 if chain_id in retrieved_chains else 0
                    name_recall_k = 1 if chain_name in retrieved_names else 0
                    # -1 guest user
                    if account_id != "-1" and chain_id not in prev_chains:
                        recall_dict[f"new_recall@{k}"] = (
                                recall_dict.get(f"new_recall@{k}", 0) + recall_k
                        )

                    else:
                        recall_dict[f"reorder_recall@{k}"] = (
                                recall_dict.get(f"reorder_recall@{k}", 0) + recall_k
                        )
                    recall_dict[f"recall@{k}"] = (
                            recall_dict.get(f"recall@{k}", 0) + recall_k
                    )
                    recall_dict[f"fair_recall@{k}"] = (
                            recall_dict.get(f"fair_recall@{k}", 0) + name_recall_k
                    )
                if chain_id not in prev_chains:
                    new_order_count += 1
                else:
                    reorder_count += 1
                orders_count += 1
            except Exception as e:
                # if account_id == "-1":
                # print(e)
                exceptions += 1

        print(
            "exceptions", exceptions, "percentage", exceptions / sample_test_orders.shape[0] * 100
        )
        print(
            "not found", not_found_errors, "percentage", not_found_errors / sample_test_orders.shape[0] * 100
        )
        errors = exceptions + not_found_errors
        for k in self.k:
            recall_k = recall_dict[f"recall@{k}"] / (
                    sample_test_orders.shape[0] - errors
            )
            reorder_recall_k = recall_dict[f"reorder_recall@{k}"] / reorder_count
            new_order_recall_k = recall_dict[f"new_recall@{k}"] / new_order_count
            fair_recall_k = recall_dict[f"fair_recall@{k}"] / (
                    sample_test_orders.shape[0] - errors
            )
            if self.logger:
                self.logger.info(f"Evaluating {evaluation_str} data")
                self.logger.info(f"recall@{k}: {recall_k}")
                self.logger.info(f"new_recall@{k}: {new_order_recall_k}")
                self.logger.info(f"reorder_recall@{k}: {reorder_recall_k}")
                self.logger.info(f"fair_recall@{k}: {fair_recall_k}")

            if mlflow.active_run():
                mlflow.log_metric(f"{evaluation_str}_recall_{k}", recall_k, step=epoch)
                mlflow.log_metric(f"{evaluation_str}_new_recall_{k}", new_order_recall_k, step=epoch)
                mlflow.log_metric(f"{evaluation_str}_reorder_recall_{k}", reorder_recall_k, step=epoch)

            print(f"Evaluating {evaluation_str} data")
            print(f"recall@{k}: {recall_k}")
            print(f"new_recall@{k}: {new_order_recall_k}")
            print(f"reorder_recall@{k}: {reorder_recall_k}")
            print(f"fair_recall@{k}: {fair_recall_k}")
            tf.summary.scalar(f"{evaluation_str}_recall@{k}", recall_k, step=epoch)
            tf.summary.scalar(f"{evaluation_str}_reorder_recall@{k}", reorder_recall_k, step=epoch)
            tf.summary.scalar(f"{evaluation_str}_new_recall@{k}", new_order_recall_k, step=epoch)
            self.recall_list[f"{evaluation_str}_recall@{k}"] = self.recall_list.get(
                f"{evaluation_str}_recall@{k}", []
            ) + [recall_k]
            self.recall_list[f"{evaluation_str}_new_recall@{k}"] = self.recall_list.get(
                f"{evaluation_str}_new_recall@{k}", []
            ) + [new_order_recall_k]
            self.recall_list[f"{evaluation_str}_reorder_recall@{k}"] = self.recall_list.get(
                f"{evaluation_str}_reorder_recall@{k}", []
            ) + [reorder_recall_k]
            self.recall_list[f"{evaluation_str}_fair_recall@{k}"] = self.recall_list.get(
                f"{evaluation_str}_fair_recall@{k}", []
            ) + [fair_recall_k]

    def get_recall_history(self):
        return self.recall_list

    def display_recommendation(self, row, top_chains,
                               order_rnk=0,
                               prev_order_rnks=None):
        prev_chains = row.user_prev_chains.split()
        prev_searches = row.prev_searches
        if self.logger:
            self.logger.info("User Prev Searches " + prev_searches)
            self.logger.info("User Prev chains " + " ".join([self.chain_to_name[chain_id]
                                                             if chain_id in self.chain_to_name
                                                             else "no embedding"
                                                             for chain_id in prev_chains]))
            if prev_order_rnks:
                self.logger.info("User Prev chains ranking " + " ".join([str(x) for x in prev_order_rnks]))
            self.logger.info("User Ordered from " + self.chain_to_name[row.chain_id])
            self.logger.info("rank of user order " + str(order_rnk))
            self.logger.info(
                "Our Recommendations " + " ".join([self.chain_to_name[chain_id] for chain_id in top_chains]),
            )
        print("User Prev Searches", prev_searches)
        print(
            "User Prev chains",
            " ".join([self.chain_to_name[chain_id]
                      if chain_id in self.chain_to_name
                      else "no embedding"
                      for chain_id in prev_chains]),
        )
        if prev_order_rnks:
            print(
                "User Prev chains ranking",
                " ".join([str(x) for x in prev_order_rnks]),
            )
        print("User Ordered from", self.chain_to_name[row.chain_id])
        print("rank of user order", order_rnk)
        print(
            "Our Recommendations",
            " ".join([self.chain_to_name[chain_id] for chain_id in top_chains]),
        )

    def evaluate_tower_on_sample(self, epoch):
        old_recall_dict = {}
        new_recall_dict = {}
        recall_dict = {}
        abs_recall_dict = {}
        orders_count = 0
        errors = 0
        iterations = 0
        # users_count = self.agg_user_chains_per_loc.shape[0]
        sample_agg_users = self.sample_agg_users
        for order in tqdm(
                sample_agg_users.itertuples(), total=sample_agg_users.shape[0]
        ):
            try:
                account_id = order.account_id
                geohash = order.geohash
                chains = order.chains
                # old_chains = self.account_prev_interactions[account_id]
                # available_chains = list(
                #     self.geo_to_chains[self.geo_to_parent_geo[geohash]]
                # )
                available_chains_w_embedding = [
                    str(x)
                    # for x in available_chains
                    for x in self.chain_embeddings.index
                    # if str(x) in self.chain_embeddings.index
                ]
                # TODO: handle the following line better
                chains = [
                    chain for chain in chains if chain in available_chains_w_embedding
                ]

                chains_embeddings = self.chain_embeddings.loc[
                    available_chains_w_embedding
                ].values
                chains_embeddings = np.array(
                    [np.array([x for x in y]) for y in chains_embeddings]
                ).squeeze()
                account_embeddings = self.user_embeddings.loc[str(account_id)].values[0]

                sim_score = get_embeddings_similarity(
                    account_embeddings, chains_embeddings
                )

                chain_order = np.argsort(sim_score)[::-1]
                for k in self.k:
                    retrieved_chains = np.array(available_chains_w_embedding)[
                        chain_order[:k]
                    ]

                    recall_k = weighted_recall(retrieved_chains, chains)
                    intersection_k = list_intersection_count(retrieved_chains, chains)
                    abs_recall_dict[f"recall@{k}"] = (
                            abs_recall_dict.get(f"recall@{k}", 0) + intersection_k
                    )
                    recall_dict[f"recall@{k}"] = (
                            recall_dict.get(f"recall@{k}", 0) + recall_k
                    )
                orders_count += len(chains)
                iterations += 1
            except Exception:
                errors += 1

        print("errors", errors)
        for k in self.k:
            recall_k = recall_dict[f"recall@{k}"] / sample_agg_users.shape[0]
            abs_recall_k = abs_recall_dict[f"recall@{k}"] / orders_count
            print(f"recall@{k}: {recall_k}")
            print(f"abs_recall@{k}: {abs_recall_k}")
            tf.summary.scalar(f"recall@{k}", recall_k, step=epoch)
