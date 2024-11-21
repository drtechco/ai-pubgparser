import os 


def new_kill_log(
        headshot: bool,
        killer: str,
        dyer: str,
        kill_method: str,
        frame_num: int
        ):
    return {
            "headshot": headshot,
            "killer": killer,
            "dyer": dyer,
            "kill_method": kill_method,
            "frame_num": frame_num
            } 

class KillLog:
    def __init__(self):
        self.headshot = False
        self.killer = ""
        self.dyer = ""
        self.frame_num = 0
        self.kill_method = ""
        self.team_pairs = []
    def to_dict(self):
        return {"headshot": self.headshot,
                "killer" : self.killer, 
                "dyer" : self.dyer,
                "frame_num" : self.frame_num,
                "kill_method": self.kill_method,
                "team_pairs": self.team_pairs
                } 

    def set_headshot(self, headshot: bool):
        self.headshot = headshot
    def set_killer(self, killer: str):
        self.killer = killer
    def set_dyer(self, dyer: str):
        self.dyer = dyer
    def set_framenum(self, frame: int):
        self.frame_num = frame
    def set_kill_method(self, km: str):
        self.kill_method = km
    def set_team_pairs(self, pair: list):
        self.team_pairs = pair
    def to_gradio_log(self, timestamp: str, metric_pair: list):
        is_dead = self.kill_method
        if is_dead == "knock":
            is_dead = "knock" 
        else:
            is_dead = "dead" 
        return {
                "TIME": timestamp,
                "KILLER": self.killer,
                "DEAD": self.dyer,
                "KILL_METHOD": self.kill_method,
                "CURRENT_STATUS": is_dead,
                "IMAGE_METRIC_PAIR": f"{metric_pair}",
                "TEAM_PAIRS": str([x["num"] for x in self.team_pairs])
            }

