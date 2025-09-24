from abstract_ranking.two_tower import TTVersion


def get_items_tt_user_online_features_type(version: TTVersion):
    if version in (TTVersion.MENUITEM_V1, ):
        from menu_item_ranking.user_features.online.features import UserOnlineFeatures
        return UserOnlineFeatures
    raise ValueError(f"Unsupported model version {version}")
