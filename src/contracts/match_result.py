class match_result:
    def __init__(self, lost_id, found_id, match_probability):
        self.lost_id = lost_id
        self.found_id = found_id
        self.match_probability = match_probability

    def to_dict(self):
        return {
            "lostId": self.lost_id,
            "foundId": self.found_id,
            "matchProbability": self.match_probability
        }
        