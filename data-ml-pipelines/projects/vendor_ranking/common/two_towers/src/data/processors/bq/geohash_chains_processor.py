from ..base_processor import BaseProcessor


class GeohashChainsProcessor(BaseProcessor):
    def process(self):
        geohash_chains_df = self.df

        geohash_chains_df = geohash_chains_df[
            ['chain_id', 'chain_name', 'address_geohash', 'vendor_id']
        ].drop_duplicates(keep='first')

        return geohash_chains_df[
            [
                'chain_id',
                'chain_name',
                'address_geohash',
                'vendor_id'
            ]
        ]