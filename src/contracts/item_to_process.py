from datetime import datetime

class item_to_process:
    def __init__(self, item_type, id, type_, color, location, date):
        self.item_type = item_type
        self.id = id
        self.type = type_
        self.color = color
        self.location = location
        self.date = date if type(date) is datetime else datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ")
    
    @classmethod
    def from_dict(cls, item_dict: dict):
        return cls(
            item_type=item_dict["item_type"], id=item_dict["id"],
            color=item_dict["color"], location=item_dict["location"],
            date=item_dict["date"], type_=item_dict["type"]
        )