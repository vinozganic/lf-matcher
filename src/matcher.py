from contracts import item_to_process, match_result

class Matcher:
    def __init__(self):
        pass

    def process_message(self, message: item_to_process) -> match_result:
        return match_result(lost_id="64204552f0c5f0bb3f216a12", found_id="6421a8be4f5e5ff4e2e91ff4", match_probability=0.5)

