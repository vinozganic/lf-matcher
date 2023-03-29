class match_result:
    def __init__(self, lost_id, found_id, match_probability):
        self.lost_id = lost_id
        self.found_id = found_id
        self.match_probability = match_probability
        