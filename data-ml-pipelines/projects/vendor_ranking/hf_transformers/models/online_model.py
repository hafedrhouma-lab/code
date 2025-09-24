import torch
import torch.nn as nn

class OnlineModel(nn.Module):
    def __init__(self, offline_output_dim, order_hour_vocab_size, order_hour_dim,
                 # order_weekday_vocab_size, order_weekday_dim,
                 area_id_vocab_size, area_id_dim, geohash_vocab_size, geohash_dim, vocab_size, dropout_rate=0.2):
        super(OnlineModel, self).__init__()

        # Embeddings for online features
        self.order_hour_embedding = nn.Embedding(order_hour_vocab_size, order_hour_dim)
        # self.order_weekday_embedding = nn.Embedding(order_weekday_vocab_size, order_weekday_dim)

        self.area_id_embedding = nn.Embedding(area_id_vocab_size, area_id_dim)
        self.geohash_embedding = nn.Embedding(geohash_vocab_size, geohash_dim)

        combined_hidden_size = offline_output_dim + order_hour_dim + area_id_dim + geohash_dim

        self.interaction_layer = nn.Sequential(
            nn.Linear(combined_hidden_size, combined_hidden_size),
            nn.BatchNorm1d(combined_hidden_size),
            nn.ReLU(),
            nn.Linear(combined_hidden_size, combined_hidden_size),
            nn.BatchNorm1d(combined_hidden_size),
            nn.ReLU(),
            nn.Linear(combined_hidden_size, combined_hidden_size),
            nn.BatchNorm1d(combined_hidden_size),
            nn.ReLU()
        )

        # Final linear layer to produce the logits
        self.final_layer = nn.Linear(combined_hidden_size, vocab_size)

        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, offline_output, order_hour, delivery_area_id, geohash, labels=None):
        device = offline_output.device

        # Embed the online features
        order_hour_emb = self.order_hour_embedding(order_hour.to(device))
        # order_weekday_emb = self.order_weekday_embedding(order_weekday.to(device))
        area_id_emb = self.area_id_embedding(delivery_area_id.to(device))
        geohash_emb = self.geohash_embedding(geohash.to(device))

        # Ensure the dimensions are compatible for concatenation
        if len(order_hour_emb.shape) == 3:
            order_hour_emb = order_hour_emb.squeeze(1)
        # if len(order_weekday_emb.shape) == 3:
        #     order_weekday_emb = order_weekday_emb.squeeze(1)
        if len(area_id_emb.shape) == 3:
            area_id_emb = area_id_emb.squeeze(1)
        if len(geohash_emb.shape) == 3:
            geohash_emb = geohash_emb.squeeze(1)

        # Concatenate the offline features with the online features
        # combined_features = torch.cat((offline_output, order_hour_emb, order_weekday_emb, area_id_emb, geohash_emb), dim=1)
        combined_features = torch.cat((offline_output, order_hour_emb, area_id_emb, geohash_emb), dim=1)

        combined_features = self.dropout(combined_features)

        combined_features = self.interaction_layer(combined_features)

        # Compute the logits
        logits = self.final_layer(combined_features).to(device)

        # Compute the loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1).to(device))

        return loss, logits
